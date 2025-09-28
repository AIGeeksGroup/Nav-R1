"""
Task-specific fine-tuning for Nav-R1 on different embodied AI benchmarks.
This module supports fine-tuning on VLN (R2R, RxR), ObjectNav (HM3D), 
embodied dialogue, planning, and reasoning tasks.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import json
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from navr1.config import NavR1Config, TaskFinetuneConfig
from navr1.models.policy import FastInSlowPolicy
from navr1.datasets.nav_cot import build_dataloader
from navr1.datasets import build_task_dataloader


class TaskSpecificHead(nn.Module):
    """Task-specific output head for different embodied AI tasks"""
    
    def __init__(self, input_dim: int, task_type: str):
        super().__init__()
        self.task_type = task_type
        
        if task_type in ["vln_r2r", "vln_rxr"]:
            # VLN tasks: output navigation actions
            self.head = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, 6)  # navigation actions
            )
        elif task_type == "objectnav_hm3d":
            # ObjectNav: output object detection + navigation
            self.head = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, 6)  # navigation actions
            )
            self.object_head = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, 1000)  # object categories
            )
        elif task_type in ["embodied_dialogue", "embodied_planning", "embodied_reasoning"]:
            # Text generation tasks
            self.head = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.ReLU(),
                nn.Linear(input_dim, 512)  # text embedding dimension
            )
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for task-specific head"""
        if self.task_type == "objectnav_hm3d":
            return {
                "action_logits": self.head(x),
                "object_logits": self.object_head(x)
            }
        else:
            return {"logits": self.head(x)}


class TaskFinetuneTrainer:
    """Task-specific fine-tuning trainer"""
    
    def __init__(self, cfg: NavR1Config, policy: FastInSlowPolicy, 
                 task_type: str, output_dir: str):
        self.cfg = cfg
        self.policy = policy
        self.task_type = task_type
        self.output_dir = output_dir
        self.device = policy.device
        
        # Get task-specific config
        task_config = getattr(cfg.task_finetune, task_type, {})
        self.learning_rate = task_config.get("learning_rate", 2e-5)
        self.batch_size = task_config.get("batch_size", 4)
        self.epochs = task_config.get("epochs", 5)
        
        # Initialize task-specific head
        self.task_head = TaskSpecificHead(cfg.model.hidden_dim, task_type).to(self.device)
        
        # Initialize optimizer
        self.optimizer = self._setup_optimizer()
        
        # Initialize loss functions
        self.loss_fn = self._setup_loss_function()
        
        # Training state
        self.global_step = 0
        self.best_metric = float('-inf')
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Task finetune trainer initialized for {task_type}")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer for task-specific fine-tuning"""
        # Get trainable parameters
        params = list(self.task_head.parameters())
        
        # Optionally fine-tune policy components
        if self.task_type in ["vln_r2r", "vln_rxr", "objectnav_hm3d"]:
            # For navigation tasks, fine-tune the entire policy
            params.extend(list(self.policy.text_encoder.parameters()))
            params.extend(list(self.policy.slow.parameters()))
            params.extend(list(self.policy.fast.parameters()))
            
            if hasattr(self.policy, 'dr1_backbone') and self.policy.dr1_backbone is not None:
                if not self.policy.cfg.freeze_3dr1_encoder:
                    params.extend(list(self.policy.dr1_backbone.parameters()))
                else:
                    params.extend(list(self.policy.dr1_backbone.scene_encoder.parameters()))
                    params.extend(list(self.policy.dr1_backbone.spatial_reasoner.parameters()))
        
        return optim.AdamW(params, lr=self.learning_rate, weight_decay=0.01)
    
    def _setup_loss_function(self):
        """Setup loss function based on task type"""
        if self.task_type in ["vln_r2r", "vln_rxr"]:
            return nn.CrossEntropyLoss()
        elif self.task_type == "objectnav_hm3d":
            return {
                "action_loss": nn.CrossEntropyLoss(),
                "object_loss": nn.CrossEntropyLoss()
            }
        elif self.task_type in ["embodied_dialogue", "embodied_planning", "embodied_reasoning"]:
            return nn.MSELoss()  # For text generation, use MSE on embeddings
        else:
            return nn.CrossEntropyLoss()
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare batch for task-specific training"""
        instructions = batch.get("instruction", [""])
        
        if self.task_type in ["vln_r2r", "vln_rxr"]:
            # VLN tasks: create dummy targets for now
            batch_size = len(instructions)
            targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            return {
                "instructions": instructions,
                "targets": targets
            }
        elif self.task_type == "objectnav_hm3d":
            # ObjectNav: create dummy targets for actions and objects
            batch_size = len(instructions)
            action_targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            object_targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
            return {
                "instructions": instructions,
                "action_targets": action_targets,
                "object_targets": object_targets
            }
        elif self.task_type in ["embodied_dialogue", "embodied_planning", "embodied_reasoning"]:
            # Text generation tasks: create dummy targets for now
            batch_size = len(instructions)
            targets = torch.zeros(batch_size, 512, device=self.device)  # text embeddings
            return {
                "instructions": instructions,
                "targets": targets
            }
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def _compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute task-specific loss"""
        # Forward pass through policy
        obs = {"batch": batch}
        policy_outputs = self.policy.forward(obs)
        slow_state = policy_outputs["slow_state"]
        
        # Forward pass through task head
        task_outputs = self.task_head(slow_state)
        
        if self.task_type in ["vln_r2r", "vln_rxr"]:
            # VLN tasks
            logits = task_outputs["logits"]
            targets = batch["targets"]
            loss = self.loss_fn(logits, targets)
            return {"total_loss": loss}
        
        elif self.task_type == "objectnav_hm3d":
            # ObjectNav tasks
            action_logits = task_outputs["action_logits"]
            object_logits = task_outputs["object_logits"]
            action_targets = batch["action_targets"]
            object_targets = batch["object_targets"]
            
            action_loss = self.loss_fn["action_loss"](action_logits, action_targets)
            object_loss = self.loss_fn["object_loss"](object_logits, object_targets)
            total_loss = action_loss + 0.5 * object_loss
            
            return {
                "action_loss": action_loss,
                "object_loss": object_loss,
                "total_loss": total_loss
            }
        
        elif self.task_type in ["embodied_dialogue", "embodied_planning", "embodied_reasoning"]:
            # Text generation tasks
            logits = task_outputs["logits"]
            targets = batch["targets"]
            loss = self.loss_fn(logits, targets)
            return {"total_loss": loss}
        
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step"""
        self.policy.train()
        self.task_head.train()
        
        # Prepare batch
        prepared_batch = self._prepare_batch(batch)
        
        # Compute loss
        losses = self._compute_loss(prepared_batch)
        total_loss = losses["total_loss"]
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.optimizer.param_groups[0]['params'] if p.requires_grad],
            max_norm=1.0
        )
        
        self.optimizer.step()
        self.global_step += 1
        
        # Return loss values
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation step"""
        self.policy.eval()
        self.task_head.eval()
        
        total_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                prepared_batch = self._prepare_batch(batch)
                losses = self._compute_loss(prepared_batch)
                
                for k, v in losses.items():
                    if k not in total_losses:
                        total_losses[k] = 0.0
                    total_losses[k] += v.item() if isinstance(v, torch.Tensor) else v
                
                num_batches += 1
        
        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save task-specific checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "task_type": self.task_type,
            "model_state_dict": {
                "text_encoder": self.policy.text_encoder.state_dict(),
                "slow": self.policy.slow.state_dict(),
                "fast": self.policy.fast.state_dict(),
                "task_head": self.task_head.state_dict(),
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        
        # Add 3D-R1 backbone state if present
        if hasattr(self.policy, 'dr1_backbone') and self.policy.dr1_backbone is not None:
            checkpoint["model_state_dict"]["dr1_backbone"] = self.policy.dr1_backbone.state_dict()
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.output_dir, f"{self.task_type}_checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.output_dir, f"best_{self.task_type}_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Best {self.task_type} model saved to {best_path}")
        
        logger.info(f"{self.task_type} checkpoint saved to {checkpoint_path}")
    
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """Main training loop for task-specific fine-tuning"""
        logger.info(f"Starting {self.task_type} fine-tuning...")
        
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            
            # Training phase
            train_losses = {}
            num_train_batches = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"{self.task_type} Epoch {epoch + 1}")
            for batch in progress_bar:
                step_losses = self.train_step(batch)
                
                # Accumulate losses
                for k, v in step_losses.items():
                    if k not in train_losses:
                        train_losses[k] = 0.0
                    train_losses[k] += v
                
                num_train_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    "loss": f"{step_losses['total_loss']:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}"
                })
            
            # Average training losses
            avg_train_losses = {k: v / num_train_batches for k, v in train_losses.items()}
            logger.info(f"Epoch {epoch + 1} - Train losses: {avg_train_losses}")
            
            # Validation phase
            if val_dataloader is not None:
                val_losses = self.validate(val_dataloader)
                logger.info(f"Epoch {epoch + 1} - Val losses: {val_losses}")
                
                # Check for best model (using negative loss as metric)
                val_metric = -val_losses["total_loss"]
                is_best = val_metric > self.best_metric
                if is_best:
                    self.best_metric = val_metric
                
                # Save best model
                self.save_checkpoint(epoch, is_best=is_best)
            else:
                # Save checkpoint without validation
                self.save_checkpoint(epoch)
        
        logger.info(f"{self.task_type} fine-tuning completed!")


def finetune_task(cfg: NavR1Config, task_type: str, output_dir: str,
                  checkpoint_path: Optional[str] = None) -> FastInSlowPolicy:
    """
    Fine-tune Nav-R1 on a specific task
    
    Args:
        cfg: Nav-R1 configuration
        task_type: Type of task to fine-tune on
        output_dir: Output directory for checkpoints
        checkpoint_path: Optional path to resume from checkpoint
        
    Returns:
        Fine-tuned policy model
    """
    # Initialize policy
    policy = FastInSlowPolicy(cfg.model)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=policy.device)
        policy.text_encoder.load_state_dict(checkpoint["model_state_dict"]["text_encoder"])
        policy.slow.load_state_dict(checkpoint["model_state_dict"]["slow"])
        policy.fast.load_state_dict(checkpoint["model_state_dict"]["fast"])
        
        if "dr1_backbone" in checkpoint["model_state_dict"] and \
           hasattr(policy, 'dr1_backbone') and policy.dr1_backbone is not None:
            policy.dr1_backbone.load_state_dict(checkpoint["model_state_dict"]["dr1_backbone"])
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    
    # Initialize trainer
    trainer = TaskFinetuneTrainer(cfg, policy, task_type, output_dir)
    
    # Build task-specific dataloaders if dataset paths are configured
    try:
        train_dataloader = build_task_dataloader(cfg, task_type, split="train", shuffle=True)
        val_dataloader = build_task_dataloader(cfg, task_type, split="val", shuffle=False)
    except Exception:
        # Fallback to generic dataloaders
        train_dataloader = build_dataloader(cfg.dataset, split="train")
        val_dataloader = build_dataloader(cfg.dataset, split="val")
    
    # Train
    trainer.train(train_dataloader, val_dataloader)
    
    return policy


# Task-specific fine-tuning functions
def finetune_vln_r2r(cfg: NavR1Config, output_dir: str, checkpoint_path: Optional[str] = None):
    """Fine-tune on VLN R2R dataset"""
    return finetune_task(cfg, "vln_r2r", output_dir, checkpoint_path)


def finetune_vln_rxr(cfg: NavR1Config, output_dir: str, checkpoint_path: Optional[str] = None):
    """Fine-tune on VLN RxR dataset"""
    return finetune_task(cfg, "vln_rxr", output_dir, checkpoint_path)


def finetune_objectnav_hm3d(cfg: NavR1Config, output_dir: str, checkpoint_path: Optional[str] = None):
    """Fine-tune on ObjectNav HM3D dataset"""
    return finetune_task(cfg, "objectnav_hm3d", output_dir, checkpoint_path)


def finetune_embodied_dialogue(cfg: NavR1Config, output_dir: str, checkpoint_path: Optional[str] = None):
    """Fine-tune on embodied dialogue tasks"""
    return finetune_task(cfg, "embodied_dialogue", output_dir, checkpoint_path)


def finetune_embodied_planning(cfg: NavR1Config, output_dir: str, checkpoint_path: Optional[str] = None):
    """Fine-tune on embodied planning tasks"""
    return finetune_task(cfg, "embodied_planning", output_dir, checkpoint_path)


def finetune_embodied_reasoning(cfg: NavR1Config, output_dir: str, checkpoint_path: Optional[str] = None):
    """Fine-tune on embodied reasoning tasks (SQA3D)"""
    return finetune_task(cfg, "embodied_reasoning", output_dir, checkpoint_path)
