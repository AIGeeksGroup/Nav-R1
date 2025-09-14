"""
Nav-R1 Policy Network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import math
from .navr1_backbone import NavR1Backbone


class PolicyHead(nn.Module):
    """
    Policy head for action prediction
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        action_dim: int = 4,  # MOVE_FORWARD, TURN_LEFT, TURN_RIGHT, STOP
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        
        # Policy network
        layers = []
        current_dim = hidden_size
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, current_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current_dim = current_dim // 2
            
        layers.append(nn.Linear(current_dim, action_dim))
        
        self.policy_net = nn.Sequential(*layers)
        
        # Value network (for RL)
        value_layers = []
        current_dim = hidden_size
        
        for i in range(num_layers - 1):
            value_layers.extend([
                nn.Linear(current_dim, current_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            current_dim = current_dim // 2
            
        value_layers.append(nn.Linear(current_dim, 1))
        
        self.value_net = nn.Sequential(*value_layers)
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through policy head
        
        Args:
            features: (batch_size, hidden_size) or (batch_size, seq_len, hidden_size)
            
        Returns:
            action_logits: (batch_size, action_dim)
            value: (batch_size, 1)
        """
        # Handle sequence input by taking mean pooling
        if len(features.shape) == 3:
            features = features.mean(dim=1)  # (batch_size, hidden_size)
            
        # Get action logits
        action_logits = self.policy_net(features)  # (batch_size, action_dim)
        
        # Get value estimate
        value = self.value_net(features)  # (batch_size, 1)
        
        return action_logits, value


class ReasoningHead(nn.Module):
    """
    Reasoning head for Chain-of-Thought generation
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        vocab_size: int = 32000,
        max_length: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Reasoning network
        self.reasoning_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size),
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through reasoning head
        
        Args:
            features: (batch_size, seq_len, hidden_size)
            
        Returns:
            reasoning_logits: (batch_size, seq_len, vocab_size)
        """
        return self.reasoning_net(features)


class NavR1Policy(nn.Module):
    """
    Complete Nav-R1 policy network
    """
    
    def __init__(
        self,
        backbone_config: Dict[str, Any],
        policy_config: Dict[str, Any],
        reasoning_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        
        # Initialize backbone
        self.backbone = NavR1Backbone(**backbone_config)
        
        # Initialize policy head
        self.policy_head = PolicyHead(**policy_config)
        
        # Initialize reasoning head if specified
        if reasoning_config is not None:
            self.reasoning_head = ReasoningHead(**reasoning_config)
        else:
            self.reasoning_head = None
            
        self.hidden_size = backbone_config["fusion_config"]["hidden_size"]
        
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        point_clouds: Optional[torch.Tensor] = None,
        return_reasoning: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Nav-R1 policy
        
        Args:
            images: (batch_size, num_images, 3, height, width)
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            image_mask: (batch_size, num_images) - 1 for valid images, 0 for padding
            point_clouds: (batch_size, num_points, 3) - optional 3D point clouds
            return_reasoning: Whether to return reasoning outputs
            
        Returns:
            outputs: Dictionary containing action_logits, value, and optionally reasoning_logits
        """
        # Get backbone features
        features = self.backbone(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_mask=image_mask,
            point_clouds=point_clouds,
        )  # (batch_size, seq_len, hidden_size)
        
        # Get policy outputs
        action_logits, value = self.policy_head(features)
        
        outputs = {
            "action_logits": action_logits,
            "value": value,
        }
        
        # Get reasoning outputs if requested and available
        if return_reasoning and self.reasoning_head is not None:
            reasoning_logits = self.reasoning_head(features)
            outputs["reasoning_logits"] = reasoning_logits
            
        return outputs
    
    def get_action_probs(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        point_clouds: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Get action probabilities
        
        Args:
            images: (batch_size, num_images, 3, height, width)
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            image_mask: (batch_size, num_images) - 1 for valid images, 0 for padding
            point_clouds: (batch_size, num_points, 3) - optional 3D point clouds
            temperature: Temperature for softmax
            
        Returns:
            action_probs: (batch_size, action_dim)
        """
        outputs = self.forward(images, input_ids, attention_mask, image_mask, point_clouds)
        action_logits = outputs["action_logits"]
        
        # Apply temperature and softmax
        action_probs = F.softmax(action_logits / temperature, dim=-1)
        
        return action_probs
    
    def sample_action(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        point_clouds: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Args:
            images: (batch_size, num_images, 3, height, width)
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            image_mask: (batch_size, num_images) - 1 for valid images, 0 for padding
            point_clouds: (batch_size, num_points, 3) - optional 3D point clouds
            temperature: Temperature for sampling
            deterministic: Whether to use deterministic sampling (argmax)
            
        Returns:
            actions: (batch_size,) - sampled action indices
            action_logits: (batch_size, action_dim)
            action_probs: (batch_size, action_dim)
        """
        outputs = self.forward(images, input_ids, attention_mask, image_mask, point_clouds)
        action_logits = outputs["action_logits"]
        value = outputs["value"]
        
        # Apply temperature
        action_logits_scaled = action_logits / temperature
        
        # Get action probabilities
        action_probs = F.softmax(action_logits_scaled, dim=-1)
        
        # Sample actions
        if deterministic:
            actions = torch.argmax(action_probs, dim=-1)
        else:
            actions = torch.multinomial(action_probs, num_samples=1).squeeze(-1)
            
        return actions, action_logits, action_probs
    
    def get_value(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        point_clouds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get value estimate
        
        Args:
            images: (batch_size, num_images, 3, height, width)
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            image_mask: (batch_size, num_images) - 1 for valid images, 0 for padding
            point_clouds: (batch_size, num_points, 3) - optional 3D point clouds
            
        Returns:
            value: (batch_size, 1)
        """
        outputs = self.forward(images, input_ids, attention_mask, image_mask, point_clouds)
        return outputs["value"]
    
    def get_reasoning(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        point_clouds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get reasoning outputs
        
        Args:
            images: (batch_size, num_images, 3, height, width)
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            image_mask: (batch_size, num_images) - 1 for valid images, 0 for padding
            point_clouds: (batch_size, num_points, 3) - optional 3D point clouds
            
        Returns:
            reasoning_logits: (batch_size, seq_len, vocab_size)
        """
        if self.reasoning_head is None:
            raise ValueError("Reasoning head not initialized")
            
        outputs = self.forward(images, input_ids, attention_mask, image_mask, point_clouds, return_reasoning=True)
        return outputs["reasoning_logits"]
