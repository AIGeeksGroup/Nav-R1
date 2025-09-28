"""
Supervised Fine-Tuning (SFT) trainer for Nav-R1 using Nav-CoT-110K dataset.
This module implements the cold-start initialization phase using supervised learning.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Iterable, Optional
from dataclasses import dataclass
import json
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from navr1.config import NavR1Config, SFTConfig
from navr1.models.policy import FastInSlowPolicy
from navr1.datasets.nav_cot import build_dataloader


class SFTLoss(nn.Module):
    """Loss function for supervised fine-tuning"""
    
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                slow_state: torch.Tensor, target_states: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute SFT loss
        
        Args:
            logits: Action logits from policy
            targets: Target action indices
            slow_state: Slow reasoning state
            target_states: Optional target reasoning states
            
        Returns:
            Dictionary of loss components
        """
        # Action prediction loss
        action_loss = self.ce_loss(logits, targets)
        
        losses = {"action_loss": action_loss}
        
        # Optional reasoning state loss
        if target_states is not None:
            reasoning_loss = self.mse_loss(slow_state, target_states)
            losses["reasoning_loss"] = reasoning_loss
        
        # Total loss
        total_loss = action_loss
        if "reasoning_loss" in losses:
            total_loss += 0.1 * losses["reasoning_loss"]  # Weight reasoning loss
        
        losses["total_loss"] = total_loss
        return losses


@dataclass
class SFTTrainer:
    """Supervised Fine-Tuning trainer for Nav-R1"""
    
    cfg: NavR1Config
    policy: FastInSlowPolicy
    output_dir: str
    
    def __post_init__(self):
        self.sft_cfg = self.cfg.sft
        self.device = self.policy.device
        
        # Initialize loss function
        self.loss_fn = SFTLoss(self.cfg.model.action_dim)
        
        # Initialize optimizer
        self.optimizer = self._setup_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._setup_scheduler()
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"SFT Trainer initialized with output directory: {self.output_dir}")
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer for SFT training"""
        # Get trainable parameters
        if self.cfg.model.use_3dr1_backbone and self.cfg.model.freeze_3dr1_encoder:
            # Only train policy components, not 3D-R1 backbone
            params = (
                list(self.policy.text_encoder.parameters()) +
                list(self.policy.slow.parameters()) +
                list(self.policy.fast.parameters())
            )
            if hasattr(self.policy, 'dr1_backbone') and self.policy.dr1_backbone is not None:
                # Add trainable parts of 3D-R1 backbone (scene_encoder, spatial_reasoner)
                params.extend(list(self.policy.dr1_backbone.scene_encoder.parameters()))
                params.extend(list(self.policy.dr1_backbone.spatial_reasoner.parameters()))
        else:
            # Train all parameters
            params = list(self.policy.text_encoder.parameters()) + \
                    list(self.policy.slow.parameters()) + \
                    list(self.policy.fast.parameters())
            if hasattr(self.policy, 'dr1_backbone') and self.policy.dr1_backbone is not None:
                params.extend(list(self.policy.dr1_backbone.parameters()))
        
        return optim.AdamW(params, lr=self.sft_cfg.learning_rate, weight_decay=0.01)
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler.LRScheduler]:
        """Setup learning rate scheduler"""
        if self.sft_cfg.warmup_steps > 0:
            return optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.sft_cfg.warmup_steps
            )
        return None
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare batch for training"""
        instructions = batch.get("instruction", [""])
        action_space = batch.get("action_space", [["forward", "left", "right", "stop"]])
        
        # Create dummy targets for now (in real implementation, these would come from dataset)
        batch_size = len(instructions)
        targets = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        return {
            "instructions": instructions,
            "action_space": action_space,
            "targets": targets
        }
    
    def _compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute loss for a batch"""
        # Forward pass
        obs = {"batch": batch}
        outputs = self.policy.forward(obs)
        
        logits = outputs["logits"]
        slow_state = outputs["slow_state"]
        targets = batch["targets"]
        
        # Compute loss
        losses = self.loss_fn(logits, targets, slow_state)
        return losses
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Single training step"""
        self.policy.train()
        
        # Prepare batch
        prepared_batch = self._prepare_batch(batch)
        
        # Compute loss
        losses = self._compute_loss(prepared_batch)
        total_loss = losses["total_loss"]
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.sft_cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.optimizer.param_groups[0]['params'] if p.requires_grad],
                self.sft_cfg.max_grad_norm
            )
        
        self.optimizer.step()
        
        # Update scheduler
        if self.scheduler is not None and self.global_step < self.sft_cfg.warmup_steps:
            self.scheduler.step()
        
        self.global_step += 1
        
        # Return loss values
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in losses.items()}
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validation step"""
        self.policy.eval()
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
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": {
                "text_encoder": self.policy.text_encoder.state_dict(),
                "slow": self.policy.slow.state_dict(),
                "fast": self.policy.fast.state_dict(),
            },
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
        }
        
        # Add 3D-R1 backbone state if present
        if hasattr(self.policy, 'dr1_backbone') and self.policy.dr1_backbone is not None:
            checkpoint["model_state_dict"]["dr1_backbone"] = self.policy.dr1_backbone.state_dict()
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.output_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        self.policy.text_encoder.load_state_dict(checkpoint["model_state_dict"]["text_encoder"])
        self.policy.slow.load_state_dict(checkpoint["model_state_dict"]["slow"])
        self.policy.fast.load_state_dict(checkpoint["model_state_dict"]["fast"])
        
        # Load 3D-R1 backbone if present
        if "dr1_backbone" in checkpoint["model_state_dict"] and \
           hasattr(self.policy, 'dr1_backbone') and self.policy.dr1_backbone is not None:
            self.policy.dr1_backbone.load_state_dict(checkpoint["model_state_dict"]["dr1_backbone"])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # Load scheduler state
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        # Load training state
        self.global_step = checkpoint.get("global_step", 0)
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def train(self, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None):
        """Main training loop"""
        logger.info("Starting SFT training...")
        
        for epoch in range(self.sft_cfg.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.sft_cfg.epochs}")
            
            # Training phase
            train_losses = {}
            num_train_batches = 0
            
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
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
                
                # Save checkpoint periodically
                if self.global_step % self.sft_cfg.save_steps == 0:
                    self.save_checkpoint(epoch)
            
            # Average training losses
            avg_train_losses = {k: v / num_train_batches for k, v in train_losses.items()}
            logger.info(f"Epoch {epoch + 1} - Train losses: {avg_train_losses}")
            
            # Validation phase
            if val_dataloader is not None:
                val_losses = self.validate(val_dataloader)
                logger.info(f"Epoch {epoch + 1} - Val losses: {val_losses}")
                
                # Check for best model
                val_loss = val_losses["total_loss"]
                is_best = val_loss < self.best_loss
                if is_best:
                    self.best_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                # Save best model
                self.save_checkpoint(epoch, is_best=is_best)
                
                # Early stopping
                if self.patience_counter >= self.cfg.training.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            else:
                # Save checkpoint without validation
                self.save_checkpoint(epoch)
        
        logger.info("SFT training completed!")


def train_sft(cfg: NavR1Config, output_dir: str, 
              checkpoint_path: Optional[str] = None) -> FastInSlowPolicy:
    """
    Train Nav-R1 using supervised fine-tuning on Nav-CoT-110K dataset
    
    Args:
        cfg: Nav-R1 configuration
        output_dir: Output directory for checkpoints
        checkpoint_path: Optional path to resume from checkpoint
        
    Returns:
        Trained policy model
    """
    # Initialize policy
    policy = FastInSlowPolicy(cfg.model)
    
    # Initialize trainer
    trainer = SFTTrainer(cfg, policy, output_dir)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        trainer.load_checkpoint(checkpoint_path)
    
    # Build dataloaders
    train_dataloader = build_dataloader(cfg.dataset, split="train")
    val_dataloader = build_dataloader(cfg.dataset, split="val")
    
    # Train
    trainer.train(train_dataloader, val_dataloader)
    
    return policy
