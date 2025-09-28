from typing import Any, Dict, Optional, List
from dataclasses import dataclass
import torch
import torch.nn as nn

from navr1.config import ModelConfig
from navr1.models.dr1_backbone import DR1Backbone


class SimpleTextEncoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.emb = nn.Embedding(1000, dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.emb(token_ids)
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return x


class FastHead(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Linear(input_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SlowReasoner(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return h[-1]


@dataclass
class FastInSlowPolicy:
    cfg: ModelConfig

    def __post_init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize 3D-R1 backbone if enabled
        if self.cfg.use_3dr1_backbone:
            self.dr1_backbone = DR1Backbone(
                model_path=self.cfg.dr1_model_path,
                freeze_encoder=self.cfg.freeze_3dr1_encoder
            ).to(self.device)
            # Use 3D-R1 hidden dimension
            backbone_dim = self.dr1_backbone.hidden_dim
        else:
            self.dr1_backbone = None
            backbone_dim = self.cfg.text_dim
        
        # Initialize policy components
        self.text_encoder = SimpleTextEncoder(self.cfg.text_dim)
        self.slow = SlowReasoner(backbone_dim, self.cfg.hidden_dim)
        self.fast = FastHead(self.cfg.hidden_dim, self.cfg.action_dim)
        
        # Move to device
        self.text_encoder.to(self.device)
        self.slow.to(self.device)
        self.fast.to(self.device)

    def tokenize(self, texts):
        max_len = 16
        batch = []
        for t in texts:
            ids = [min(ord(c) % 1000, 999) for c in t][:max_len]
            ids += [0] * (max_len - len(ids))
            batch.append(ids)
        return torch.tensor(batch, dtype=torch.long, device=self.device)

    def encode_instruction(self, instructions: List[str]) -> torch.Tensor:
        """Encode instructions using either 3D-R1 backbone or simple text encoder"""
        if self.dr1_backbone is not None:
            # Use 3D-R1 backbone for scene understanding
            dr1_output = self.dr1_backbone(instructions)
            return dr1_output["scene_embeddings"]
        else:
            # Use simple text encoder
            tok = self.tokenize(instructions)
            return self.text_encoder(tok)
    
    @torch.no_grad()
    def act(self, obs: Dict[str, Any]):
        instructions = obs.get("batch", {}).get("instruction", [""])
        
        # Encode instructions
        encoded_instructions = self.encode_instruction(instructions)
        
        # Process through slow reasoning and fast action head
        seq = encoded_instructions.unsqueeze(1)
        slow_state = self.slow(seq)
        logits = self.fast(slow_state)
        action_idx = torch.argmax(logits, dim=-1).item()
        
        return ["forward", "left", "right", "back", "turn_left", "stop"][action_idx % 6]
    
    def forward(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass for training"""
        instructions = obs.get("batch", {}).get("instruction", [""])
        
        # Encode instructions
        encoded_instructions = self.encode_instruction(instructions)
        
        # Process through slow reasoning and fast action head
        seq = encoded_instructions.unsqueeze(1)
        slow_state = self.slow(seq)
        logits = self.fast(slow_state)
        
        return {
            "logits": logits,
            "slow_state": slow_state,
            "encoded_instructions": encoded_instructions
        }
