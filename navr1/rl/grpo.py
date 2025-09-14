"""
Group Relative Policy Optimization (GRPO) for Nav-R1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
import wandb
from tqdm import tqdm
import os
import json
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from accelerate import Accelerator
import math

from ..models.policy import NavR1Policy
from ..simulators.habitat import HabitatSimulator, HabitatStubSimulator
from ..utils.distributed import (
    setup_distributed, cleanup_distributed, is_main_process, get_rank, get_world_size,
    create_ddp_model, save_checkpoint_on_main_process, load_checkpoint_on_main_process,
    DistributedMetrics, setup_logging, get_optimizer_state_dict, load_optimizer_state_dict
)


class GRPOTrainer:
    """
    Group Relative Policy Optimization trainer for Nav-R1
    """
    
    def __init__(
        self,
        model: NavR1Policy,
        simulator: HabitatSimulator,
        config: Dict[str, Any] = None,
        device: str = "cuda",
        rank: int = 0,
        world_size: int = 1,
        use_ddp: bool = False,
    ):
        self.model = model
        self.simulator = simulator
        self.config = config or {}
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.use_ddp = use_ddp
        
        # RL configuration
        self.learning_rate = self.config.get("learning_rate", 3e-6)
        self.num_epochs = self.config.get("num_epochs", 5)
        self.batch_size = self.config.get("batch_size", 4)
        self.rollout_length = self.config.get("rollout_length", 128)
        self.gamma = self.config.get("gamma", 0.99)
        self.lambda_gae = self.config.get("lambda_gae", 0.95)
        self.clip_ratio = self.config.get("clip_ratio", 0.2)
        self.value_loss_coef = self.config.get("value_loss_coef", 0.5)
        self.entropy_coef = self.config.get("entropy_coef", 0.01)
        self.max_grad_norm = self.config.get("max_grad_norm", 0.5)
        
        # Reward configuration
        self.reward_config = self.config.get("rewards", {})
        self.format_weight = self.reward_config.get("format_weight", 1.0)
        self.understanding_weight = self.reward_config.get("understanding_weight", 2.0)
        self.navigation_weight = self.reward_config.get("navigation_weight", 3.0)
        
        # Fast-in-Slow reasoning configuration
        self.fast_slow_config = self.config.get("fast_slow", {})
        self.slow_reasoning_interval = self.fast_slow_config.get("slow_reasoning_interval", 5)
        self.fast_reasoning_steps = self.fast_slow_config.get("fast_reasoning_steps", 3)
        self.reasoning_cache_size = self.fast_slow_config.get("reasoning_cache_size", 100)
        
        # Setup distributed training if enabled
        if self.use_ddp:
            setup_distributed(self.rank, self.world_size)
            self.device = f"cuda:{self.rank}"
            self.model = self.model.to(self.device)
            self.model = create_ddp_model(self.model, device_ids=[self.rank])
        else:
            # Initialize accelerator for single GPU training
            self.accelerator = Accelerator(
                mixed_precision="fp16" if self.config.get("mixed_precision", True) else "no",
            )
            self.model = self.accelerator.prepare(self.model)
        
        # Initialize optimizer
        if self.use_ddp:
            model_params = self.model.module.parameters()
        else:
            model_params = self.model.parameters()
            
        self.optimizer = AdamW(
            model_params,
            lr=self.learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_reward = float('-inf')
        
        # Reasoning cache for Fast-in-Slow reasoning
        self.reasoning_cache = {}
        
        # Distributed metrics
        self.distributed_metrics = DistributedMetrics()
        
        # Logging
        self.use_wandb = self.config.get("use_wandb", False)
        if self.use_wandb and is_main_process():
            wandb.init(
                project=self.config.get("wandb_project", "navr1"),
                name=self.config.get("experiment_name", "grpo_experiment"),
                config=self.config,
            )
    
    def collect_rollouts(self, num_episodes: int = 10) -> Dict[str, List]:
        """Collect rollouts from the environment"""
        rollouts = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "dones": [],
            "episode_infos": [],
        }
        
        self.model.eval()
        
        for episode_idx in range(num_episodes):
            episode_data = self._run_episode()
            
            # Add episode data to rollouts
            for key in rollouts:
                if key in episode_data:
                    rollouts[key].extend(episode_data[key])
        
        return rollouts
    
    def _run_episode(self) -> Dict[str, List]:
        """Run a single episode"""
        episode_data = {
            "observations": [],
            "actions": [],
            "rewards": [],
            "values": [],
            "log_probs": [],
            "dones": [],
            "episode_infos": [],
        }
        
        # Reset environment
        obs = self.simulator.reset()
        episode_info = self.simulator.get_episode_info()
        
        # Initialize reasoning cache for this episode
        episode_reasoning_cache = {}
        
        step_count = 0
        done = False
        
        while not done and step_count < self.rollout_length:
            # Fast-in-Slow reasoning
            if step_count % self.slow_reasoning_interval == 0:
                # Slow reasoning: generate detailed reasoning
                reasoning_output = self._slow_reasoning(obs, episode_info)
                episode_reasoning_cache[step_count] = reasoning_output
            else:
                # Fast reasoning: use cached reasoning or quick inference
                reasoning_output = self._fast_reasoning(obs, episode_info, episode_reasoning_cache)
            
            # Get action from model
            action, value, log_prob = self._get_action(obs, reasoning_output)
            
            # Execute action
            next_obs, reward, done, info = self.simulator.step(action)
            
            # Calculate composite reward
            composite_reward = self._calculate_composite_reward(
                reward, reasoning_output, action, info
            )
            
            # Store data
            episode_data["observations"].append(obs)
            episode_data["actions"].append(action)
            episode_data["rewards"].append(composite_reward)
            episode_data["values"].append(value)
            episode_data["log_probs"].append(log_prob)
            episode_data["dones"].append(done)
            episode_data["episode_infos"].append(episode_info)
            
            # Update for next step
            obs = next_obs
            episode_info = info
            step_count += 1
        
        return episode_data
    
    def _slow_reasoning(self, obs: Dict[str, Any], episode_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed reasoning (slow)"""
        # This is where the model would generate detailed Chain-of-Thought reasoning
        # For now, we'll create a simplified version
        
        reasoning_output = {
            "observation": "I can see the current environment",
            "reasoning_steps": [
                "Analyzing the current position and orientation",
                "Understanding the instruction and goals",
                "Planning the next steps based on the environment",
            ],
            "conclusion": "I should take the next action based on my analysis",
            "confidence": 0.8,
        }
        
        return reasoning_output
    
    def _fast_reasoning(
        self,
        obs: Dict[str, Any],
        episode_info: Dict[str, Any],
        reasoning_cache: Dict[int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Use cached reasoning or quick inference (fast)"""
        # Use the most recent reasoning from cache
        if reasoning_cache:
            latest_reasoning = max(reasoning_cache.keys())
            return reasoning_cache[latest_reasoning]
        
        # Fallback to quick reasoning
        return {
            "observation": "Quick observation",
            "reasoning_steps": ["Quick analysis"],
            "conclusion": "Quick decision",
            "confidence": 0.6,
        }
    
    def _get_action(
        self,
        obs: Dict[str, Any],
        reasoning_output: Dict[str, Any]
    ) -> Tuple[str, float, float]:
        """Get action from model"""
        # Convert observation to model input format
        images = obs["rgb"].unsqueeze(0)  # Add batch dimension
        input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=self.device)  # Dummy input
        attention_mask = torch.ones_like(input_ids)
        image_mask = torch.ones(1, images.shape[1], device=self.device)
        
        # Get model outputs
        with torch.no_grad():
            outputs = self.model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_mask=image_mask,
            )
            
            action_logits = outputs["action_logits"]
            value = outputs["value"]
            
            # Sample action
            action_probs = F.softmax(action_logits, dim=-1)
            action_dist = torch.distributions.Categorical(action_probs)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            # Convert action index to action string
            action_mapping = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]
            action_str = action_mapping[action.item()]
            
            return action_str, value.item(), log_prob.item()
    
    def _calculate_composite_reward(
        self,
        base_reward: float,
        reasoning_output: Dict[str, Any],
        action: str,
        info: Dict[str, Any]
    ) -> float:
        """Calculate composite reward with format, understanding, and navigation components"""
        # Format reward: reward for following proper reasoning format
        format_reward = self._calculate_format_reward(reasoning_output)
        
        # Understanding reward: reward for semantic understanding
        understanding_reward = self._calculate_understanding_reward(reasoning_output, info)
        
        # Navigation reward: reward for navigation performance
        navigation_reward = self._calculate_navigation_reward(base_reward, action, info)
        
        # Composite reward
        composite_reward = (
            self.format_weight * format_reward +
            self.understanding_weight * understanding_reward +
            self.navigation_weight * navigation_reward
        )
        
        return composite_reward
    
    def _calculate_format_reward(self, reasoning_output: Dict[str, Any]) -> float:
        """Calculate format reward for proper reasoning structure"""
        # Reward for having proper reasoning structure
        format_score = 0.0
        
        if "observation" in reasoning_output:
            format_score += 0.2
        if "reasoning_steps" in reasoning_output and len(reasoning_output["reasoning_steps"]) > 0:
            format_score += 0.3
        if "conclusion" in reasoning_output:
            format_score += 0.2
        if "confidence" in reasoning_output:
            format_score += 0.1
        
        return format_score
    
    def _calculate_understanding_reward(
        self,
        reasoning_output: Dict[str, Any],
        info: Dict[str, Any]
    ) -> float:
        """Calculate understanding reward for semantic grounding"""
        # This would involve more sophisticated semantic analysis
        # For now, we'll use a simplified version
        
        understanding_score = 0.0
        
        # Reward for confidence
        if "confidence" in reasoning_output:
            understanding_score += reasoning_output["confidence"] * 0.5
        
        # Reward for reasoning quality (simplified)
        if "reasoning_steps" in reasoning_output:
            understanding_score += min(len(reasoning_output["reasoning_steps"]) * 0.1, 0.3)
        
        return understanding_score
    
    def _calculate_navigation_reward(
        self,
        base_reward: float,
        action: str,
        info: Dict[str, Any]
    ) -> float:
        """Calculate navigation reward for path fidelity"""
        navigation_score = base_reward
        
        # Additional rewards for navigation performance
        if info.get("success", False):
            navigation_score += 10.0
        
        if info.get("collision", False):
            navigation_score -= 1.0
        
        # Reward for progress (simplified)
        if action == "MOVE_FORWARD":
            navigation_score += 0.1
        
        return navigation_score
    
    def compute_advantages(self, rollouts: Dict[str, List]) -> List[float]:
        """Compute advantages using GAE"""
        rewards = rollouts["rewards"]
        values = rollouts["values"]
        dones = rollouts["dones"]
        
        advantages = []
        returns = []
        
        # Compute returns and advantages
        next_value = 0.0
        next_advantage = 0.0
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_value = 0.0
                next_advantage = 0.0
            
            # Compute return
            return_val = rewards[i] + self.gamma * next_value
            returns.insert(0, return_val)
            
            # Compute advantage using GAE
            delta = rewards[i] + self.gamma * next_value - values[i]
            advantage = delta + self.gamma * self.lambda_gae * next_advantage
            advantages.insert(0, advantage)
            
            next_value = values[i]
            next_advantage = advantage
        
        return advantages, returns
    
    def train_step(self, rollouts: Dict[str, List]) -> Dict[str, float]:
        """Train the model on collected rollouts"""
        self.model.train()
        
        # Compute advantages
        advantages, returns = self.compute_advantages(rollouts)
        
        # Convert to tensors
        observations = rollouts["observations"]
        actions = rollouts["actions"]
        old_log_probs = torch.tensor(rollouts["log_probs"], device=self.device)
        advantages = torch.tensor(advantages, device=self.device)
        returns = torch.tensor(returns, device=self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute policy loss
        policy_loss = 0.0
        value_loss = 0.0
        entropy_loss = 0.0
        
        # Process in batches
        batch_size = min(self.batch_size, len(observations))
        num_batches = len(observations) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            
            batch_obs = observations[start_idx:end_idx]
            batch_actions = actions[start_idx:end_idx]
            batch_old_log_probs = old_log_probs[start_idx:end_idx]
            batch_advantages = advantages[start_idx:end_idx]
            batch_returns = returns[start_idx:end_idx]
            
            # Get current policy outputs
            current_outputs = self._get_batch_outputs(batch_obs)
            current_log_probs = current_outputs["log_probs"]
            current_values = current_outputs["values"]
            
            # Compute policy loss (PPO-style)
            ratio = torch.exp(current_log_probs - batch_old_log_probs)
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
            policy_loss += -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            value_loss += F.mse_loss(current_values, batch_returns)
            
            # Compute entropy loss
            entropy_loss += -current_outputs["entropy"].mean()
        
        # Average losses
        policy_loss /= num_batches
        value_loss /= num_batches
        entropy_loss /= num_batches
        
        # Total loss
        total_loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
        }
    
    def _get_batch_outputs(self, batch_obs: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Get model outputs for a batch of observations"""
        # This is a simplified implementation
        # In practice, you would properly batch the observations
        
        batch_size = len(batch_obs)
        
        # Create dummy inputs (in practice, you would process the actual observations)
        images = torch.randn(batch_size, 8, 3, 224, 224, device=self.device)
        input_ids = torch.randint(0, 1000, (batch_size, 512), device=self.device)
        attention_mask = torch.ones_like(input_ids)
        image_mask = torch.ones(batch_size, 8, device=self.device)
        
        # Get model outputs
        outputs = self.model(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_mask=image_mask,
        )
        
        # Compute log probabilities and entropy
        action_logits = outputs["action_logits"]
        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)
        
        log_probs = action_dist.log_prob(torch.randint(0, 4, (batch_size,), device=self.device))
        entropy = action_dist.entropy()
        
        return {
            "log_probs": log_probs,
            "values": outputs["value"].squeeze(-1),
            "entropy": entropy,
        }
    
    def train(self, num_episodes: int = 100) -> Dict[str, float]:
        """Train the model using GRPO"""
        print("Starting GRPO training...")
        
        for epoch in range(self.num_epochs):
            self.epoch = epoch
            
            # Collect rollouts
            print(f"Collecting rollouts for epoch {epoch}...")
            rollouts = self.collect_rollouts(num_episodes)
            
            # Train on rollouts
            print(f"Training on rollouts for epoch {epoch}...")
            train_metrics = self.train_step(rollouts)
            
            # Log metrics
            self._log_metrics(train_metrics, "train")
            
            # Evaluate
            eval_metrics = self.evaluate(num_episodes=10)
            self._log_metrics(eval_metrics, "eval")
            
            # Save checkpoint
            if eval_metrics.get("avg_reward", 0) > self.best_reward:
                self.best_reward = eval_metrics["avg_reward"]
                self.save_checkpoint(is_best=True)
            
            self.save_checkpoint()
        
        print("GRPO training completed!")
        
        # Save final model
        self.save_checkpoint(is_final=True)
        
        return train_metrics
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the model"""
        self.model.eval()
        
        total_rewards = []
        success_rates = []
        episode_lengths = []
        
        for episode_idx in range(num_episodes):
            episode_reward = 0.0
            episode_length = 0
            episode_success = False
            
            # Reset environment
            obs = self.simulator.reset()
            episode_info = self.simulator.get_episode_info()
            
            done = False
            while not done and episode_length < self.rollout_length:
                # Get action
                action, _, _ = self._get_action(obs, {})
                
                # Execute action
                next_obs, reward, done, info = self.simulator.step(action)
                
                episode_reward += reward
                episode_length += 1
                episode_success = info.get("success", False)
                
                obs = next_obs
            
            total_rewards.append(episode_reward)
            success_rates.append(1.0 if episode_success else 0.0)
            episode_lengths.append(episode_length)
        
        return {
            "avg_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "success_rate": np.mean(success_rates),
            "avg_episode_length": np.mean(episode_lengths),
        }
    
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
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_reward": self.best_reward,
            "config": self.config,
        }
        
        # Save checkpoint
        if is_best:
            checkpoint_path = os.path.join(save_dir, "best_grpo_model.pt")
        elif is_final:
            checkpoint_path = os.path.join(save_dir, "final_grpo_model.pt")
        else:
            checkpoint_path = os.path.join(save_dir, f"grpo_checkpoint_step_{self.global_step}.pt")
            
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
        
        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            load_optimizer_state_dict(self.optimizer, checkpoint["optimizer_state_dict"])
            
        # Load training state
        self.global_step = checkpoint.get("global_step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        self.best_reward = checkpoint.get("best_reward", float('-inf'))
        
        if is_main_process():
            print(f"Loaded checkpoint from {checkpoint_path}")
    
    def cleanup(self):
        """Clean up distributed training resources"""
        if self.use_ddp:
            cleanup_distributed()
