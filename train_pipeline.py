"""
Complete training pipeline for Nav-R1 multi-stage training:
1. 3D-R1 backbone initialization
2. Nav-CoT-110K SFT (cold-start)
3. GRPO RL training
4. Task-specific fine-tuning
"""

import os
import sys
import yaml
import click
from rich.console import Console
from loguru import logger
from pathlib import Path

from navr1.config import NavR1Config, load_config
from navr1.datasets.nav_cot import build_dataloader
from navr1.simulators import build_simulator
from navr1.models.policy import FastInSlowPolicy
from navr1.training.sft_trainer import train_sft
from navr1.training.task_finetune import (
    finetune_vln_r2r, finetune_vln_rxr, finetune_objectnav_hm3d,
    finetune_embodied_dialogue, finetune_embodied_planning, finetune_embodied_reasoning
)
from navr1.rl.grpo import GRPOTrainer

console = Console()


def run_sft_stage(cfg: NavR1Config, workdir: str) -> str:
    """Run SFT (Supervised Fine-Tuning) stage"""
    console.rule("[bold green]Stage 1: SFT Training on Nav-CoT-110K")
    
    sft_output_dir = os.path.join(workdir, "sft")
    os.makedirs(sft_output_dir, exist_ok=True)
    
    logger.info("Starting SFT training...")
    policy = train_sft(cfg, sft_output_dir)
    
    # Return path to best SFT model
    best_sft_path = os.path.join(sft_output_dir, "best_model.pt")
    if os.path.exists(best_sft_path):
        logger.info(f"SFT training completed. Best model saved to: {best_sft_path}")
        return best_sft_path
    else:
        logger.warning("SFT training completed but best model not found")
        return os.path.join(sft_output_dir, "checkpoint_epoch_0.pt")


def run_grpo_stage(cfg: NavR1Config, workdir: str, sft_checkpoint_path: str) -> str:
    """Run GRPO RL training stage"""
    console.rule("[bold blue]Stage 2: GRPO RL Training")
    
    grpo_output_dir = os.path.join(workdir, "grpo")
    os.makedirs(grpo_output_dir, exist_ok=True)
    
    # Initialize policy and simulator
    policy = FastInSlowPolicy(cfg.model)
    sim = build_simulator(cfg.simulator)
    
    # Initialize GRPO trainer
    trainer = GRPOTrainer(cfg.rl, policy, sim, cfg, grpo_output_dir)
    
    # Load SFT checkpoint
    trainer.load_sft_checkpoint(sft_checkpoint_path)
    
    # Build dataloader for RL training
    dataloader = build_dataloader(cfg.dataset)
    
    # Train
    logger.info("Starting GRPO RL training...")
    trainer.train(dataloader)
    
    # Return path to best GRPO model
    best_grpo_path = os.path.join(grpo_output_dir, "best_grpo_model.pt")
    if os.path.exists(best_grpo_path):
        logger.info(f"GRPO training completed. Best model saved to: {best_grpo_path}")
        return best_grpo_path
    else:
        logger.warning("GRPO training completed but best model not found")
        return os.path.join(grpo_output_dir, "grpo_checkpoint_epoch_0.pt")


def run_task_finetune_stage(cfg: NavR1Config, workdir: str, grpo_checkpoint_path: str, 
                           tasks: list) -> dict:
    """Run task-specific fine-tuning stage"""
    console.rule("[bold magenta]Stage 3: Task-Specific Fine-tuning")
    
    task_results = {}
    
    for task in tasks:
        console.print(f"[bold yellow]Fine-tuning on {task}...")
        
        task_output_dir = os.path.join(workdir, f"finetune_{task}")
        os.makedirs(task_output_dir, exist_ok=True)
        
        try:
            if task == "vln_r2r":
                policy = finetune_vln_r2r(cfg, task_output_dir, grpo_checkpoint_path)
            elif task == "vln_rxr":
                policy = finetune_vln_rxr(cfg, task_output_dir, grpo_checkpoint_path)
            elif task == "objectnav_hm3d":
                policy = finetune_objectnav_hm3d(cfg, task_output_dir, grpo_checkpoint_path)
            elif task == "embodied_dialogue":
                policy = finetune_embodied_dialogue(cfg, task_output_dir, grpo_checkpoint_path)
            elif task == "embodied_planning":
                policy = finetune_embodied_planning(cfg, task_output_dir, grpo_checkpoint_path)
            elif task == "embodied_reasoning":
                policy = finetune_embodied_reasoning(cfg, task_output_dir, grpo_checkpoint_path)
            else:
                logger.error(f"Unknown task: {task}")
                continue
            
            # Save path to best model
            best_model_path = os.path.join(task_output_dir, f"best_{task}_model.pt")
            task_results[task] = {
                "success": True,
                "model_path": best_model_path,
                "output_dir": task_output_dir
            }
            
            logger.info(f"Task {task} fine-tuning completed successfully")
            
        except Exception as e:
            logger.error(f"Task {task} fine-tuning failed: {e}")
            task_results[task] = {
                "success": False,
                "error": str(e),
                "output_dir": task_output_dir
            }
    
    return task_results


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True), 
              default="navr1/configs/default.yaml", help="Path to config file")
@click.option("--workdir", type=click.Path(), default="./runs/navr1_pipeline", 
              help="Working directory for training outputs")
@click.option("--stages", type=click.Choice(["sft", "grpo", "finetune", "all"]), 
              default="all", help="Training stages to run")
@click.option("--tasks", type=str, default="vln_r2r,vln_rxr,objectnav_hm3d,embodied_dialogue,embodied_planning,embodied_reasoning",
              help="Comma-separated list of tasks for fine-tuning")
@click.option("--sft-checkpoint", type=click.Path(), default=None,
              help="Path to SFT checkpoint to resume from")
@click.option("--grpo-checkpoint", type=click.Path(), default=None,
              help="Path to GRPO checkpoint to resume from")
@click.option("--seed", type=int, default=42, help="Random seed")
@click.option("--rl-task", type=click.Choice(["auto", "vln_r2r", "vln_rxr", "objectnav_hm3d"]), default="auto",
              help="Which task's Habitat config to use during GRPO training (auto uses cfg.simulator.habitat_config)")
def main(config_path: str, workdir: str, stages: str, tasks: str, 
         sft_checkpoint: str, grpo_checkpoint: str, seed: int, rl_task: str):
    """
    Nav-R1 Multi-Stage Training Pipeline
    
    This script runs the complete training pipeline for Nav-R1:
    1. SFT training on Nav-CoT-110K dataset
    2. GRPO RL training
    3. Task-specific fine-tuning on various benchmarks
    """
    
    # Set random seed
    import torch
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load configuration
    cfg: NavR1Config = load_config(config_path)
    os.makedirs(workdir, exist_ok=True)
    
    # Setup logging
    logger.add(os.path.join(workdir, "pipeline.log"))
    logger.info(f"Nav-R1 Training Pipeline started with config: {config_path}")
    logger.info(f"Working directory: {workdir}")
    logger.info(f"Stages to run: {stages}")
    
    # Parse tasks
    task_list = [task.strip() for task in tasks.split(",")]
    logger.info(f"Tasks for fine-tuning: {task_list}")
    
    # Stage 1: SFT Training
    sft_checkpoint_path = sft_checkpoint
    if stages in ["sft", "all"]:
        if sft_checkpoint_path is None:
            sft_checkpoint_path = run_sft_stage(cfg, workdir)
        else:
            logger.info(f"Using provided SFT checkpoint: {sft_checkpoint_path}")
    
    # Stage 2: GRPO RL Training
    grpo_checkpoint_path = grpo_checkpoint
    if stages in ["grpo", "all"]:
        if grpo_checkpoint_path is None:
            if sft_checkpoint_path is None:
                logger.error("SFT checkpoint is required for GRPO training")
                return
            # If user specified a target task for RL env, switch Habitat YAML accordingly
            if rl_task in ["vln_r2r", "vln_rxr", "objectnav_hm3d"]:
                task_cfg = getattr(cfg.task_finetune, rl_task, {}) if hasattr(cfg, 'task_finetune') else {}
                hcfg = task_cfg.get("habitat_config") if isinstance(task_cfg, dict) else None
                if hcfg:
                    logger.info(f"Using task-specific Habitat config for RL: {hcfg}")
                    cfg.simulator.habitat_config = hcfg
                else:
                    logger.warning(f"No habitat_config found under task_finetune.{rl_task}. Using cfg.simulator.habitat_config: {cfg.simulator.habitat_config}")
            grpo_checkpoint_path = run_grpo_stage(cfg, workdir, sft_checkpoint_path)
        else:
            logger.info(f"Using provided GRPO checkpoint: {grpo_checkpoint_path}")
    
    # Stage 3: Task-Specific Fine-tuning
    if stages in ["finetune", "all"]:
        if grpo_checkpoint_path is None:
            logger.error("GRPO checkpoint is required for task fine-tuning")
            return
        
        task_results = run_task_finetune_stage(cfg, workdir, grpo_checkpoint_path, task_list)
        
        # Print summary
        console.rule("[bold green]Training Pipeline Summary")
        console.print(f"Working directory: {workdir}")
        console.print(f"SFT checkpoint: {sft_checkpoint_path}")
        console.print(f"GRPO checkpoint: {grpo_checkpoint_path}")
        
        console.print("\n[bold]Task Fine-tuning Results:")
        for task, result in task_results.items():
            if result["success"]:
                console.print(f"  ✅ {task}: {result['model_path']}")
            else:
                console.print(f"  ❌ {task}: {result['error']}")
    
    # Save final configuration
    final_config_path = os.path.join(workdir, "final_config.yaml")
    with open(final_config_path, "w") as f:
        yaml.dump(cfg.dict(), f, default_flow_style=False)
    
    console.print(f"\n[bold green]Training pipeline completed!")
    console.print(f"All outputs saved to: {workdir}")
    console.print(f"Final config saved to: {final_config_path}")


if __name__ == "__main__":
    main()
