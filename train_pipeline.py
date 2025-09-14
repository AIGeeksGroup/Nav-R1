#!/usr/bin/env python3
"""
Complete training pipeline for Nav-R1
"""

import argparse
import os
import yaml
import torch
from typing import Dict, Any
import json
from datetime import datetime

from navr1.models.policy import NavR1Policy
from navr1.datasets.nav_cot import create_nav_cot_dataset
from navr1.datasets import create_embodied_dataset, create_embodied_dataloader
from navr1.training.sft_trainer import SFTTrainer
from navr1.rl.grpo import GRPOTrainer
from navr1.simulators.habitat import HabitatSimulator, HabitatStubSimulator


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict[str, Any]) -> NavR1Policy:
    """Create Nav-R1 model from configuration"""
    model_config = config["model"]
    
    # Vision encoder config
    vision_config = {
        "model_name": model_config["vision_encoder"]["type"],
        "pretrained_path": model_config["vision_encoder"].get("pretrained_path"),
        "freeze_vision": model_config["vision_encoder"].get("freeze_vision", False),
        "image_size": config["dataset"]["image_size"],
        "hidden_size": model_config["multimodal_fusion"]["hidden_size"],
    }
    
    # Language encoder config
    language_config = {
        "model_name": model_config["language_model"]["type"],
        "pretrained_path": model_config["language_model"].get("pretrained_path"),
        "freeze_lm": model_config["language_model"].get("freeze_lm", False),
        "hidden_size": model_config["multimodal_fusion"]["hidden_size"],
    }
    
    # Fusion config
    fusion_config = model_config["multimodal_fusion"]
    
    # Policy config
    policy_config = model_config["policy_head"]
    
    # Reasoning config (optional)
    reasoning_config = None
    if "reasoning_head" in model_config:
        reasoning_config = model_config["reasoning_head"]
    
    # Create backbone config
    backbone_config = {
        "vision_config": vision_config,
        "language_config": language_config,
        "fusion_config": fusion_config,
    }
    
    # Create model
    model = NavR1Policy(
        backbone_config=backbone_config,
        policy_config=policy_config,
        reasoning_config=reasoning_config,
    )
    
    return model


def create_simulator(config: Dict[str, Any]) -> HabitatSimulator:
    """Create simulator from configuration"""
    simulator_config = config["simulator"]
    
    try:
        simulator = HabitatSimulator(
            config_path=simulator_config["habitat_config"],
            scene_dataset_path=simulator_config["scene_dataset"],
            episode_dataset_path=simulator_config["episode_dataset"],
            max_episode_steps=simulator_config["max_episode_steps"],
            success_reward=simulator_config["success_reward"],
            step_penalty=simulator_config["step_penalty"],
            collision_penalty=simulator_config["collision_penalty"],
            device=config["hardware"]["device"],
        )
    except ImportError:
        print("Warning: Habitat not available, using stub simulator")
        simulator = HabitatStubSimulator(
            config_path=simulator_config["habitat_config"],
            scene_dataset_path=simulator_config["scene_dataset"],
            episode_dataset_path=simulator_config["episode_dataset"],
            max_episode_steps=simulator_config["max_episode_steps"],
            success_reward=simulator_config["success_reward"],
            step_penalty=simulator_config["step_penalty"],
            collision_penalty=simulator_config["collision_penalty"],
            device=config["hardware"]["device"],
        )
    
    return simulator


def stage1_sft_training(config: Dict[str, Any], workdir: str) -> str:
    """Stage 1: Supervised Fine-Tuning on Nav-CoT-110K"""
    print("=" * 60)
    print("Stage 1: Supervised Fine-Tuning (SFT)")
    print("=" * 60)
    
    # Create model
    model = create_model(config)
    
    # Create datasets
    train_dataset = create_nav_cot_dataset(
        data_path=config["dataset"]["path"],
        split="train",
        max_sequence_length=config["dataset"]["max_sequence_length"],
        max_images=config["dataset"]["max_images"],
        image_size=config["dataset"]["image_size"],
        tokenizer_name=config["dataset"]["tokenizer"]["type"],
    )
    
    val_dataset = create_nav_cot_dataset(
        data_path=config["dataset"]["path"],
        split="val",
        max_sequence_length=config["dataset"]["max_sequence_length"],
        max_images=config["dataset"]["max_images"],
        image_size=config["dataset"]["image_size"],
        tokenizer_name=config["dataset"]["tokenizer"]["type"],
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config["training"],
        device=config["hardware"]["device"],
    )
    
    # Train
    trainer.train()
    
    # Save checkpoint
    checkpoint_path = os.path.join(workdir, "sft_checkpoint.pt")
    trainer.save_checkpoint(is_final=True)
    
    print(f"SFT training completed! Checkpoint saved to {checkpoint_path}")
    return checkpoint_path




def stage3_embodied_finetune(config: Dict[str, Any], workdir: str, rl_checkpoint: str) -> str:
    """Stage 3: Embodied Task Fine-tuning on RL weights"""
    print("=" * 60)
    print("Stage 3: Embodied Task Fine-tuning (on RL weights)")
    print("=" * 60)
    
    # Create model and load RL checkpoint
    model = create_model(config)
    if os.path.exists(rl_checkpoint):
        checkpoint = torch.load(rl_checkpoint, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded RL checkpoint from {rl_checkpoint}")
    
    # Create embodied task datasets
    embodied_task = config.get("embodied_task", "dialogue")
    task_config = config["embodied_tasks"][embodied_task]
    
    train_dataset = create_embodied_dataset(
        task_type=task_config["dataset_name"],
        data_path=task_config["data_path"],
        split="train",
        max_sequence_length=task_config["max_sequence_length"],
        max_images=task_config["max_images"],
        image_size=task_config["image_size"],
    )
    
    val_dataset = create_embodied_dataset(
        task_type=task_config["dataset_name"],
        data_path=task_config["data_path"],
        split="val",
        max_sequence_length=task_config["max_sequence_length"],
        max_images=task_config["max_images"],
        image_size=task_config["image_size"],
    )
    
    # Create trainer
    trainer_config = config["training"].copy()
    trainer_config["task_type"] = embodied_task
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=trainer_config,
        device=config["hardware"]["device"],
    )
    
    # Train
    trainer.train()
    
    # Save checkpoint
    checkpoint_path = os.path.join(workdir, f"embodied_finetune_{embodied_task}_checkpoint.pt")
    trainer.save_checkpoint(is_final=True)
    
    print(f"Embodied task fine-tuning completed! Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def stage2_rl_training(config: Dict[str, Any], workdir: str, sft_checkpoint: str) -> str:
    """Stage 2: Reinforcement Learning with GRPO"""
    print("=" * 60)
    print("Stage 2: Reinforcement Learning (GRPO)")
    print("=" * 60)
    
    # Create model and load SFT checkpoint
    model = create_model(config)
    if os.path.exists(sft_checkpoint):
        checkpoint = torch.load(sft_checkpoint, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded SFT checkpoint from {sft_checkpoint}")
    
    # Create simulator
    simulator = create_simulator(config)
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        simulator=simulator,
        config=config["rl"],
        device=config["hardware"]["device"],
    )
    
    # Train
    trainer.train()
    
    # Save checkpoint
    checkpoint_path = os.path.join(workdir, "rl_checkpoint.pt")
    trainer.save_checkpoint(is_final=True)
    
    print(f"RL training completed! Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def run_complete_pipeline(config: Dict[str, Any], workdir: str):
    """Run the complete training pipeline"""
    print("Starting Nav-R1 Complete Training Pipeline")
    print("=" * 60)
    
    # Create work directory
    os.makedirs(workdir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(workdir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Training log
    training_log = {
        "start_time": datetime.now().isoformat(),
        "stages": [],
        "checkpoints": {},
    }
    
    try:
        # Stage 1: SFT Training
        sft_checkpoint = stage1_sft_training(config, workdir)
        training_log["stages"].append("sft_completed")
        training_log["checkpoints"]["sft"] = sft_checkpoint
        
        # Stage 2: RL Training
        rl_checkpoint = stage2_rl_training(config, workdir, sft_checkpoint)
        training_log["stages"].append("rl_completed")
        training_log["checkpoints"]["rl"] = rl_checkpoint
        
        # Stage 3: Embodied Task Fine-tuning (on RL weights)
        embodied_checkpoint = stage3_embodied_finetune(config, workdir, rl_checkpoint)
        training_log["stages"].append("embodied_finetune_completed")
        training_log["checkpoints"]["embodied_finetune"] = embodied_checkpoint
        
        # Final evaluation
        print("=" * 60)
        print("Final Evaluation")
        print("=" * 60)
        
        # Load final model
        model = create_model(config)
        checkpoint = torch.load(embodied_checkpoint, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        
        # Create simulator for evaluation
        simulator = create_simulator(config)
        
        # Evaluate
        from evaluate import evaluate
        results = evaluate(
            model=model,
            simulator=simulator,
            num_episodes=config["evaluation"]["num_episodes"],
            save_videos=config["evaluation"]["save_videos"],
            video_dir=os.path.join(workdir, "videos"),
        )
        
        # Save evaluation results
        eval_path = os.path.join(workdir, "evaluation_results.json")
        with open(eval_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("Final evaluation completed!")
        print(f"Results saved to {eval_path}")
        
        # Close simulator
        simulator.close()
        
        # Update training log
        training_log["end_time"] = datetime.now().isoformat()
        training_log["final_evaluation"] = results["metrics"]
        training_log["status"] = "completed"
        
    except Exception as e:
        print(f"Training pipeline failed: {e}")
        training_log["end_time"] = datetime.now().isoformat()
        training_log["status"] = "failed"
        training_log["error"] = str(e)
        raise
    
    finally:
        # Save training log
        log_path = os.path.join(workdir, "training_log.json")
        with open(log_path, 'w') as f:
            json.dump(training_log, f, indent=2)
        
        print(f"Training log saved to {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Run complete Nav-R1 training pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--workdir", type=str, default="runs/navr1_pipeline", help="Working directory")
    parser.add_argument("--stage", type=str, choices=["sft", "embodied_finetune", "rl", "all"], 
                       default="all", help="Training stage to run")
    parser.add_argument("--embodied_task", type=str, choices=["dialogue", "reasoning", "planning", "vln", "objectnav"], 
                       default="dialogue", help="Embodied task type")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = config["hardware"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
        config["hardware"]["device"] = device
    
    # Set random seeds
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Add embodied task to config
    config["embodied_task"] = args.embodied_task
    
    # Run training
    if args.stage == "all":
        run_complete_pipeline(config, args.workdir)
    else:
        # Run specific stage
        if args.stage == "sft":
            stage1_sft_training(config, args.workdir)
        elif args.stage == "embodied_finetune":
            rl_checkpoint = os.path.join(args.workdir, "rl_checkpoint.pt")
            stage3_embodied_finetune(config, args.workdir, rl_checkpoint)
        elif args.stage == "rl":
            sft_checkpoint = os.path.join(args.workdir, "sft_checkpoint.pt")
            stage2_rl_training(config, args.workdir, sft_checkpoint)


if __name__ == "__main__":
    main()
