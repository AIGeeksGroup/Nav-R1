"""
Supervised Fine-Tuning (SFT) Trainer for Nav-R1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, get_linear_schedule_with_warmup
import wandb
from tqdm import tqdm
import os
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from accelerate import Accelerator
from transformers import get_linear_schedule_with_warmup

from ..models.policy import NavR1Policy
from ..datasets.nav_cot import NavCoTDataset, NavCoTDataLoader
from ..utils.distributed import (
    setup_distributed, cleanup_distributed, is_main_process, get_rank, get_world_size,
    create_ddp_model, create_distributed_sampler, save_checkpoint_on_main_process,
    load_checkpoint_on_main_process, broadcast_checkpoint, DistributedMetrics,
    setup_logging, get_optimizer_state_dict, load_optimizer_state_dict
)


class SFTTrainer:
    """
    Supervised Fine-Tuning trainer for Nav-R1
    """
    
    def __init__(
        self,
        model: NavR1Policy,
        train_dataset: NavCoTDataset,
        val_dataset: Optional[NavCoTDataset] = None,
        config: Dict[str, Any] = None,
        device: str = "cuda",
        rank: int = 0,
        world_size: int = 1,
        use_ddp: bool = False,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config or {}
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.use_ddp = use_ddp
        
        # Training configuration
        self.batch_size = self.config.get("batch_size", 8)
        self.learning_rate = self.config.get("learning_rate", 1e-5)
        self.num_epochs = self.config.get("num_epochs", 10)
        self.warmup_steps = self.config.get("warmup_steps", 1000)
        self.weight_decay = self.config.get("weight_decay", 0.01)
        self.gradient_accumulation_steps = self.config.get("gradient_accumulation_steps", 4)
        self.max_grad_norm = self.config.get("max_grad_norm", 1.0)
        self.save_steps = self.config.get("save_steps", 1000)
        self.eval_steps = self.config.get("eval_steps", 500)
        self.logging_steps = self.config.get("logging_steps", 100)
        
        # Setup distributed training if enabled
        if self.use_ddp:
            setup_distributed(self.rank, self.world_size)
            self.device = f"cuda:{self.rank}"
            self.model = self.model.to(self.device)
            self.model = create_ddp_model(self.model, device_ids=[self.rank])
        else:
            # Initialize accelerator for single GPU training
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                mixed_precision="fp16" if self.config.get("mixed_precision", True) else "no",
            )
            self.model = self.accelerator.prepare(self.model)
        
        # Initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        self._setup_optimizer()
        
        # Initialize data loaders
        self.train_loader = None
        self.val_loader = None
        self._setup_data_loaders()
        
        # Loss functions
        self.action_loss_fn = nn.CrossEntropyLoss()
        self.reasoning_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Distributed metrics
        self.distributed_metrics = DistributedMetrics()
        
        # Logging
        self.use_wandb = self.config.get("use_wandb", False)
        if self.use_wandb and is_main_process():
            wandb.init(
                project=self.config.get("wandb_project", "navr1"),
                name=self.config.get("experiment_name", "sft_experiment"),
                config=self.config,
            )
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # Get model parameters
        if self.use_ddp:
            model = self.model.module
        else:
            model = self.model
        
        # Setup optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Calculate total training steps
        if self.use_ddp:
            # For distributed training, adjust batch size by world size
            effective_batch_size = self.batch_size * self.world_size
        else:
            effective_batch_size = self.batch_size
            
        total_steps = len(self.train_dataset) // (effective_batch_size * self.gradient_accumulation_steps) * self.num_epochs
        
        # Setup scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps,
        )
    
    def _setup_data_loaders(self):
        """Setup data loaders"""
        if self.use_ddp:
            # Use distributed samplers for DDP
            train_sampler = create_distributed_sampler(self.train_dataset, shuffle=True)
            val_sampler = create_distributed_sampler(self.val_dataset, shuffle=False) if self.val_dataset else None
            
            # Train loader
            train_loader = NavCoTDataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                sampler=train_sampler,
                num_workers=self.config.get("dataloader_num_workers", 4),
                pin_memory=self.config.get("pin_memory", True),
            ).get_dataloader()
            
            self.train_loader = train_loader
            
            # Validation loader
            if self.val_dataset is not None:
                val_loader = NavCoTDataLoader(
                    dataset=self.val_dataset,
                    batch_size=self.batch_size,
                    sampler=val_sampler,
                    num_workers=self.config.get("dataloader_num_workers", 4),
                    pin_memory=self.config.get("pin_memory", True),
                ).get_dataloader()
                
                self.val_loader = val_loader
        else:
            # Use accelerator for single GPU training
            # Train loader
            train_loader = NavCoTDataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.config.get("dataloader_num_workers", 4),
                pin_memory=self.config.get("pin_memory", True),
            ).get_dataloader()
            
            self.train_loader = self.accelerator.prepare(train_loader)
            
            # Validation loader
            if self.val_dataset is not None:
                val_loader = NavCoTDataLoader(
                    dataset=self.val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.config.get("dataloader_num_workers", 4),
                    pin_memory=self.config.get("pin_memory", True),
                ).get_dataloader()
                
                self.val_loader = self.accelerator.prepare(val_loader)
    
    def _compute_loss(
        self,
        batch: Dict[str, Any],
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss"""
        losses = {}
        
        # Action prediction loss
        action_logits = outputs["action_logits"]
        # Convert action space to indices (simplified for demo)
        action_targets = self._get_action_targets(batch["action_space"])
        action_loss = self.action_loss_fn(action_logits, action_targets)
        losses["action_loss"] = action_loss
        
        # Reasoning loss (if available)
        if "reasoning_logits" in outputs and batch.get("cot_reasoning"):
            reasoning_logits = outputs["reasoning_logits"]
            reasoning_targets = self._get_reasoning_targets(batch["cot_reasoning"])
            reasoning_loss = self.reasoning_loss_fn(
                reasoning_logits.view(-1, reasoning_logits.size(-1)),
                reasoning_targets.view(-1)
            )
            losses["reasoning_loss"] = reasoning_loss
        
        # Total loss
        total_loss = losses["action_loss"]
        if "reasoning_loss" in losses:
            total_loss += 0.1 * losses["reasoning_loss"]  # Weight reasoning loss
        losses["total_loss"] = total_loss
        
        return losses
    
    def _get_action_targets(self, action_spaces: List[List[str]]) -> torch.Tensor:
        """Convert action spaces to target indices"""
        # Simplified action mapping
        action_mapping = {
            "MOVE_FORWARD": 0,
            "TURN_LEFT": 1,
            "TURN_RIGHT": 2,
            "STOP": 3,
        }
        
        targets = []
        for action_space in action_spaces:
            # For simplicity, use the first action in the space
            if action_space and action_space[0] in action_mapping:
                targets.append(action_mapping[action_space[0]])
            else:
                targets.append(0)  # Default to MOVE_FORWARD
                
        return torch.tensor(targets, device=self.device)
    
    def _get_reasoning_targets(self, cot_reasonings: List[str]) -> torch.Tensor:
        """Convert CoT reasoning to target tokens"""
        # This is a simplified implementation
        # In practice, you would tokenize the reasoning text properly
        targets = []
        for reasoning in cot_reasonings:
            # For now, create dummy targets
            # In practice, you would tokenize the reasoning text
            target_tokens = torch.randint(0, 1000, (512,))  # Dummy tokens
            targets.append(target_tokens)
            
        return torch.stack(targets).to(self.device)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        # Set epoch for distributed sampler
        if self.use_ddp and hasattr(self.train_loader.sampler, 'set_epoch'):
            self.train_loader.sampler.set_epoch(self.epoch)
        
        total_losses = {}
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch}",
            disable=not is_main_process(),
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            if self.use_ddp:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(
                images=batch["images"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                image_mask=batch["image_mask"],
                return_reasoning=True,
            )
            
            # Compute loss
            losses = self._compute_loss(batch, outputs)
            
            # Backward pass
            if self.use_ddp:
                losses["total_loss"].backward()
            else:
                self.accelerator.backward(losses["total_loss"])
            
            # Gradient clipping
            if self.max_grad_norm > 0:
                if self.use_ddp:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                else:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Update parameters
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    self._log_metrics(losses, "train")
                
                # Evaluation
                if self.val_loader is not None and self.global_step % self.eval_steps == 0:
                    val_metrics = self.evaluate()
                    self._log_metrics(val_metrics, "val")
                    
                    # Save best model
                    if val_metrics["val_total_loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["val_total_loss"]
                        self.save_checkpoint(is_best=True)
                
                # Save checkpoint
                if self.global_step % self.save_steps == 0:
                    self.save_checkpoint()
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0.0
                total_losses[key] += value.item()
            num_batches += 1
            
            # Update progress bar
            if is_main_process():
                progress_bar.set_postfix({
                    "loss": f"{losses['total_loss'].item():.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}",
                })
        
        # Average losses
        for key in total_losses:
            total_losses[key] /= num_batches
            
        return total_losses
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set"""
        if self.val_loader is None:
            return {}
            
        self.model.eval()
        total_losses = {}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", disable=not is_main_process()):
                # Move batch to device
                if self.use_ddp:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    images=batch["images"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    image_mask=batch["image_mask"],
                    return_reasoning=True,
                )
                
                # Compute loss
                losses = self._compute_loss(batch, outputs)
                
                # Accumulate losses
                for key, value in losses.items():
                    if key not in total_losses:
                        total_losses[key] = 0.0
                    total_losses[key] += value.item()
                num_batches += 1
        
        # Average losses
        for key in total_losses:
            total_losses[key] = total_losses[key] / num_batches
            total_losses[f"val_{key}"] = total_losses.pop(key)
            
        return total_losses
    
    def train(self) -> Dict[str, float]:
        """Train the model"""
        print("Starting SFT training...")
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Log epoch metrics
            self._log_metrics(train_metrics, "train_epoch")
            
            # Final evaluation
            if self.val_loader is not None:
                val_metrics = self.evaluate()
                self._log_metrics(val_metrics, "val_epoch")
        
        print("SFT training completed!")
        
        # Save final model
        self.save_checkpoint(is_final=True)
        
        return train_metrics
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics to wandb and console"""
        if not is_main_process():
            return
            
        # Add prefix to metrics
        if prefix:
            metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}
            
        # Log to wandb
        if self.use_wandb:
            wandb.log(metrics, step=self.global_step)
            
        # Log to console
        print(f"Step {self.global_step}: {metrics}")
    
    def save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint"""
        if not is_main_process():
            return
            
        save_dir = self.config.get("save_dir", "checkpoints")
        os.makedirs(save_dir, exist_ok=True)
        
        # Get model state dict
        if self.use_ddp:
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.accelerator.unwrap_model(self.model).state_dict()
        
        # Prepare checkpoint
        checkpoint = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": get_optimizer_state_dict(self.optimizer),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_val_loss": self.best_val_loss,
            "config": self.config,
        }
        
        # Save checkpoint
        if is_best:
            checkpoint_path = os.path.join(save_dir, "best_model.pt")
        elif is_final:
            checkpoint_path = os.path.join(save_dir, "final_model.pt")
        else:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_step_{self.global_step}.pt")
            
        save_checkpoint_on_main_process(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = load_checkpoint_on_main_process(checkpoint_path)
        if checkpoint is None:
            return
            
        # Load model state
        if self.use_ddp:
            self.model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model_state_dict"])
        
        # Load optimizer and scheduler state
        if "optimizer_state_dict" in checkpoint:
            load_optimizer_state_dict(self.optimizer, checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            
        # Load training state
        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        
        if is_main_process():
            print(f"Loaded checkpoint from {checkpoint_path}")
    
    def cleanup(self):
        """Clean up distributed training resources"""
        if self.use_ddp:
            cleanup_distributed()
