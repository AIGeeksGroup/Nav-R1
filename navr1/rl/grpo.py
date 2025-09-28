from typing import Any, Dict, Iterable, Optional
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import os
from loguru import logger

from navr1.config import RLConfig, NavR1Config


class RewardHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class GRPOTrainer:
    cfg: RLConfig
    policy: Any
    simulator: Any
    full_cfg: Optional[NavR1Config] = None
    output_dir: str = "./runs/grpo"

    def __post_init__(self):
        # Get trainable parameters
        params = list(self.policy.text_encoder.parameters()) + \
                list(self.policy.slow.parameters()) + \
                list(self.policy.fast.parameters())
        
        # Add 3D-R1 backbone parameters if not frozen
        if hasattr(self.policy, 'dr1_backbone') and self.policy.dr1_backbone is not None:
            if not self.policy.cfg.freeze_3dr1_encoder:
                params.extend(list(self.policy.dr1_backbone.parameters()))
            else:
                # Add only trainable parts (scene_encoder, spatial_reasoner)
                params.extend(list(self.policy.dr1_backbone.scene_encoder.parameters()))
                params.extend(list(self.policy.dr1_backbone.spatial_reasoner.parameters()))
        
        self.optimizer = optim.Adam(params, lr=self.cfg.learning_rate)
        self.value_fn = RewardHead(self.policy.cfg.hidden_dim).to(self.policy.device)
        self.value_opt = optim.Adam(self.value_fn.parameters(), lr=self.cfg.learning_rate)
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    def load_sft_checkpoint(self, checkpoint_path: str):
        """Load SFT checkpoint to initialize RL training"""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"SFT checkpoint not found at {checkpoint_path}, starting from scratch")
            return
        
        checkpoint = torch.load(checkpoint_path, map_location=self.policy.device)
        
        # Load model state
        self.policy.text_encoder.load_state_dict(checkpoint["model_state_dict"]["text_encoder"])
        self.policy.slow.load_state_dict(checkpoint["model_state_dict"]["slow"])
        self.policy.fast.load_state_dict(checkpoint["model_state_dict"]["fast"])
        
        # Load 3D-R1 backbone if present
        if "dr1_backbone" in checkpoint["model_state_dict"] and \
           hasattr(self.policy, 'dr1_backbone') and self.policy.dr1_backbone is not None:
            self.policy.dr1_backbone.load_state_dict(checkpoint["model_state_dict"]["dr1_backbone"])
        
        logger.info(f"Loaded SFT checkpoint from {checkpoint_path}")

    def cold_start(self, dataloader: Iterable[Dict[str, Any]]):
        """Cold start training (deprecated, use load_sft_checkpoint instead)"""
        logger.warning("cold_start is deprecated, use load_sft_checkpoint instead")
        self.policy.text_encoder.train()
        self.policy.slow.train()
        self.policy.fast.train()
        for _ in range(1):
            for batch in dataloader:
                tok = self.policy.tokenize(batch.get("instruction", [""]))
                txt = self.policy.text_encoder(tok)
                seq = txt.unsqueeze(1)
                slow_state = self.policy.slow(seq)
                logits = self.policy.fast(slow_state)
                targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
                loss = self.ce(logits, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                break

    def compute_rewards(self, info: Dict[str, Any]) -> torch.Tensor:
        batch_size = 1
        format_r = torch.full((batch_size,), self.cfg.format_coef)
        understanding_r = torch.full((batch_size,), self.cfg.understanding_coef)
        navigation_r = torch.full((batch_size,), self.cfg.navigation_coef if info.get("success", False) else 0.0)
        return (format_r + understanding_r + navigation_r).to(self.policy.device)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save GRPO checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "episode_count": self.episode_count,
            "model_state_dict": {
                "text_encoder": self.policy.text_encoder.state_dict(),
                "slow": self.policy.slow.state_dict(),
                "fast": self.policy.fast.state_dict(),
            },
            "value_fn_state_dict": self.value_fn.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "value_opt_state_dict": self.value_opt.state_dict(),
        }
        
        # Add 3D-R1 backbone state if present
        if hasattr(self.policy, 'dr1_backbone') and self.policy.dr1_backbone is not None:
            checkpoint["model_state_dict"]["dr1_backbone"] = self.policy.dr1_backbone.state_dict()
        
        # Save checkpoint
        checkpoint_path = os.path.join(self.output_dir, f"grpo_checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.output_dir, "best_grpo_model.pt")
            torch.save(checkpoint, best_path)
            logger.info(f"Best GRPO model saved to {best_path}")
        
        logger.info(f"GRPO checkpoint saved to {checkpoint_path}")

    def train(self, dataloader: Iterable[Dict[str, Any]]):
        """Main GRPO training loop"""
        logger.info("Starting GRPO training...")
        
        self.policy.text_encoder.train()
        self.policy.slow.train()
        self.policy.fast.train()
        
        best_reward = float('-inf')
        
        for epoch in range(self.cfg.epochs):
            logger.info(f"GRPO Epoch {epoch + 1}/{self.cfg.epochs}")
            
            epoch_rewards = []
            epoch_successes = 0
            num_episodes = 0
            
            for batch in dataloader:
                obs = self.simulator.reset(batch)
                done = False
                returns = []
                values = []
                logprobs = []
                actions = []
                infos = []
                steps = 0
                
                # Rollout phase
                while not done and steps < self.cfg.rollout_length:
                    # Get action from policy
                    instructions = batch.get("instruction", [""])
                    encoded_instructions = self.policy.encode_instruction(instructions)
                    seq = encoded_instructions.unsqueeze(1)
                    slow_state = self.policy.slow(seq)
                    logits = self.policy.fast(slow_state)
                    
                    # Sample action
                    dist = torch.distributions.Categorical(logits=logits)
                    action_idx = dist.sample()
                    action = ["forward", "left", "right", "back", "turn_left", "stop"][action_idx.item() % 6]
                    
                    # Environment step
                    obs, reward, done, info = self.simulator.step(action)
                    
                    # Compute rewards
                    r = self.compute_rewards(info)
                    v = self.value_fn(slow_state).squeeze(-1)
                    
                    # Store experience
                    returns.append(r)
                    values.append(v)
                    logprobs.append(dist.log_prob(action_idx))
                    actions.append(action_idx)
                    infos.append(info)
                    steps += 1
                
                # Compute advantages and returns
                R = torch.zeros(1, device=self.policy.device)
                gae = torch.zeros(1, device=self.policy.device)
                policy_loss = 0.0
                value_loss = 0.0
                entropy_loss = 0.0
                
                for t in reversed(range(len(returns))):
                    delta = returns[t] + self.cfg.gamma * (values[t + 1] if t + 1 < len(values) else 0) - values[t]
                    gae = delta + self.cfg.gamma * self.cfg.lam * gae
                    policy_loss = policy_loss - (logprobs[t] * gae.detach())
                    value_loss = value_loss + self.mse(values[t], returns[t].detach())
                    entropy_loss = entropy_loss - (logprobs[t] * 0.0)
                
                # Update models
                total_loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                self.optimizer.zero_grad()
                self.value_opt.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.value_opt.step()
                
                # Track metrics
                episode_reward = sum([r.item() for r in returns])
                episode_success = any([info.get("success", False) for info in infos])
                
                epoch_rewards.append(episode_reward)
                if episode_success:
                    epoch_successes += 1
                num_episodes += 1
                self.episode_count += 1
                self.global_step += 1
                
                # Log progress
                if num_episodes % 10 == 0:
                    avg_reward = sum(epoch_rewards[-10:]) / min(10, len(epoch_rewards))
                    success_rate = epoch_successes / num_episodes
                    logger.info(f"Episode {num_episodes} - Avg Reward: {avg_reward:.3f}, Success Rate: {success_rate:.3f}")
            
            # Epoch summary
            avg_epoch_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
            epoch_success_rate = epoch_successes / num_episodes if num_episodes > 0 else 0
            
            logger.info(f"Epoch {epoch + 1} - Avg Reward: {avg_epoch_reward:.3f}, Success Rate: {epoch_success_rate:.3f}")
            
            # Save checkpoint
            is_best = avg_epoch_reward > best_reward
            if is_best:
                best_reward = avg_epoch_reward
            
            self.save_checkpoint(epoch, is_best=is_best)
        
        logger.info("GRPO training completed!")
