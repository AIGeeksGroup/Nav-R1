#!/usr/bin/env python3
"""
Training script for Nav-R1
"""

import argparse
import os
import yaml
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from typing import Dict, Any

from navr1.models.policy import NavR1Policy
from navr1.datasets.nav_cot import create_nav_cot_dataset
from navr1.datasets import create_3d_dataset, create_3d_dataloader, create_embodied_dataset, create_embodied_dataloader
from navr1.training.sft_trainer import SFTTrainer
from navr1.rl.grpo import GRPOTrainer
from navr1.simulators.habitat import HabitatSimulator, HabitatStubSimulator
from navr1.utils.distributed import run_distributed_training, setup_logging


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


def train_sft(config: Dict[str, Any], workdir: str, rank: int = 0, world_size: int = 1, use_ddp: bool = False):
    """Train using Supervised Fine-Tuning"""
    if rank == 0:
        print("Starting SFT training...")
    
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
        rank=rank,
        world_size=world_size,
        use_ddp=use_ddp,
    )
    
    try:
        # Train
        trainer.train()
        
        if rank == 0:
            print("SFT training completed!")
    finally:
        # Cleanup
        trainer.cleanup()


def train_rl(config: Dict[str, Any], workdir: str, rank: int = 0, world_size: int = 1, use_ddp: bool = False):
    """Train using Reinforcement Learning (GRPO)"""
    if rank == 0:
        print("Starting RL training...")
    
    # Create model
    model = create_model(config)
    
    # Create simulator
    simulator = create_simulator(config)
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        simulator=simulator,
        config=config["rl"],
        device=config["hardware"]["device"],
        rank=rank,
        world_size=world_size,
        use_ddp=use_ddp,
    )
    
    try:
        # Train
        trainer.train()
        
        if rank == 0:
            print("RL training completed!")
    finally:
        # Cleanup
        trainer.cleanup()




def train_3d_scene_tasks(config: Dict[str, Any], workdir: str, task_name: str):
    """Train on 3D scene understanding tasks"""
    print(f"Starting 3D scene understanding training for {task_name}...")
    
    # Create model
    model = create_model(config)
    
    # Get task configuration
    task_config = config["tasks"][task_name]
    
    # Create 3D datasets
    train_dataset = create_3d_dataset(
        dataset_name=task_config["dataset_name"],
        data_path=task_config["data_path"],
        split="train",
        max_sequence_length=task_config["max_sequence_length"],
        max_points=task_config["max_points"],
        image_size=task_config["image_size"],
    )
    
    val_dataset = create_3d_dataset(
        dataset_name=task_config["dataset_name"],
        data_path=task_config["data_path"],
        split="val",
        max_sequence_length=task_config["max_sequence_length"],
        max_points=task_config["max_points"],
        image_size=task_config["image_size"],
    )
    
    # Create trainer
    trainer_config = config["training"].copy()
    trainer_config["task_type"] = task_name
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=trainer_config,
        device=config["hardware"]["device"],
    )
    
    # Train
    trainer.train()
    
    print(f"3D scene understanding training for {task_name} completed!")


def train_embodied_tasks(config: Dict[str, Any], workdir: str, task_name: str, resume_checkpoint: str = None):
    """Train on embodied tasks (dialogue, reasoning, planning, navigation)"""
    print(f"Starting embodied task training for {task_name}...")
    
    # Create model
    model = create_model(config)
    
    # Load checkpoint if provided
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        checkpoint = torch.load(resume_checkpoint, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {resume_checkpoint}")
    elif resume_checkpoint:
        print(f"Warning: Checkpoint {resume_checkpoint} not found, training from scratch")
    
    # Get task configuration
    task_config = config["embodied_tasks"][task_name]
    
    # Create embodied datasets
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
    trainer_config["task_type"] = task_name
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=trainer_config,
        device=config["hardware"]["device"],
    )
    
    # Train
    trainer.train()
    
    print(f"Embodied task training for {task_name} completed!")


def main():
    parser = argparse.ArgumentParser(description="Train Nav-R1 model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--workdir", type=str, default="runs/navr1", help="Working directory")
    parser.add_argument("--mode", type=str, choices=["sft", "rl", "3d_scene", "embodied"], 
                       default="sft", help="Training mode")
    parser.add_argument("--task_type", type=str, choices=["vln", "objectnav"], 
                       default="vln", help="Task type for task-specific fine-tuning")
    parser.add_argument("--3d_task", type=str, choices=["scanrefer", "scanqa", "nr3d", "scene30k"], 
                       default="scanrefer", help="3D scene understanding task")
    parser.add_argument("--embodied_task", type=str, choices=["dialogue", "reasoning", "planning", "vln", "objectnav"], 
                       default="dialogue", help="Embodied task type")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use for training")
    parser.add_argument("--use_ddp", action="store_true", help="Use DistributedDataParallel for multi-GPU training")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create work directory
    os.makedirs(args.workdir, exist_ok=True)
    
    # Set device
    device = config["hardware"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
        config["hardware"]["device"] = device
        args.use_ddp = False  # Disable DDP if no CUDA
    
    # Set random seeds
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Determine number of GPUs
    if args.use_ddp and torch.cuda.is_available():
        world_size = min(args.num_gpus, torch.cuda.device_count())
        if world_size > 1:
            print(f"Using {world_size} GPUs for distributed training")
            # Run distributed training
            run_distributed_training(
                train_fn=_distributed_train_wrapper,
                world_size=world_size,
                config=config,
                workdir=args.workdir,
                mode=args.mode,
                task_type=args.task_type,
                embodied_task=args.embodied_task,
                resume=args.resume,
            )
        else:
            # Single GPU training
            _train_single_gpu(config, args.workdir, args.mode, args.task_type, args.embodied_task, args.resume)
    else:
        # Single GPU training
        _train_single_gpu(config, args.workdir, args.mode, args.task_type, args.embodied_task, args.resume)


def _distributed_train_wrapper(rank: int, world_size: int, backend: str, kwargs: dict):
    """Wrapper function for distributed training"""
    config = kwargs["config"]
    workdir = kwargs["workdir"]
    mode = kwargs["mode"]
    task_type = kwargs["task_type"]
    embodied_task = kwargs["embodied_task"]
    resume = kwargs["resume"]
    
    # Setup logging for this process
    setup_logging(rank)
    
    # Train based on mode
    if mode == "sft":
        train_sft(config, workdir, rank, world_size, use_ddp=True)
    elif mode == "rl":
        train_rl(config, workdir, rank, world_size, use_ddp=True)
    elif mode == "3d_scene":
        train_3d_scene_tasks(config, workdir, task_type, rank, world_size, use_ddp=True)
    elif mode == "embodied":
        train_embodied_tasks(config, workdir, embodied_task, resume)
    else:
        raise ValueError(f"Unknown training mode: {mode}")


def _train_single_gpu(config: dict, workdir: str, mode: str, task_type: str, embodied_task: str, resume: str):
    """Train on single GPU"""
    # Train based on mode
    if mode == "sft":
        train_sft(config, workdir)
    elif mode == "rl":
        train_rl(config, workdir)
    elif mode == "3d_scene":
        train_3d_scene_tasks(config, workdir, task_type)
    elif mode == "embodied":
        train_embodied_tasks(config, workdir, embodied_task, resume)
    else:
        raise ValueError(f"Unknown training mode: {mode}")


if __name__ == "__main__":
    main()
