"""
Nav-R1 Backbone Model Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import math
from transformers import AutoModel, AutoTokenizer, AutoConfig
import timm
from einops import rearrange, repeat


class VisionEncoder(nn.Module):
    """
    3D Vision encoder based on 3D-R1 architecture
    """
    
    def __init__(
        self,
        model_name: str = "3d_r1_vision",
        pretrained_path: Optional[str] = None,
        freeze_vision: bool = False,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_size: int = 4096,
        point_cloud_dim: int = 3,
        max_points: int = 8192,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.point_cloud_dim = point_cloud_dim
        self.max_points = max_points
        
        # 3D Point Cloud Encoder (PointNet++)
        self.point_encoder = self._build_point_encoder()
        
        # 2D Image Encoder (ViT-based)
        self.image_encoder = self._build_image_encoder()
        
        # 3D Feature Fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.point_encoder.output_dim + self.image_encoder.output_dim, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size * 2, hidden_size),
        )
        
        # Freeze vision encoder if specified
        if freeze_vision:
            for param in self.parameters():
                param.requires_grad = False
                
        # Load pretrained weights if provided
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)
    
    def _build_point_encoder(self):
        """Build 3D point cloud encoder"""
        try:
            # Try to import PointNet++ from 3D-R1
            from third_party.pointnet2 import PointNet2
            point_encoder = PointNet2(
                input_dim=self.point_cloud_dim,
                output_dim=512,
                max_points=self.max_points
            )
            point_encoder.output_dim = 512
        except ImportError:
            # Fallback to simple MLP
            point_encoder = nn.Sequential(
                nn.Linear(self.point_cloud_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
            )
            point_encoder.output_dim = 512
        
        return point_encoder
    
    def _build_image_encoder(self):
        """Build 2D image encoder"""
        # Use CLIP or ViT for 2D images
        try:
            import clip
            model, preprocess = clip.load("ViT-B/32", device="cpu")
            image_encoder = model.visual
            image_encoder.output_dim = 512
        except ImportError:
            # Fallback to standard ViT
            image_encoder = timm.create_model(
                "vit_base_patch16_224",
                pretrained=True,
                num_classes=0,
                global_pool="",
            )
            image_encoder.output_dim = image_encoder.embed_dim
        
        return image_encoder
    
    def load_pretrained_weights(self, pretrained_path: str):
        """Load pretrained vision encoder weights"""
        try:
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            self.load_state_dict(checkpoint, strict=False)
            print(f"Loaded pretrained vision weights from {pretrained_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights from {pretrained_path}: {e}")
    
    def forward(self, images: torch.Tensor, point_clouds: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through 3D vision encoder
        
        Args:
            images: (batch_size, num_images, 3, height, width)
            point_clouds: (batch_size, num_points, 3) - optional 3D point clouds
            
        Returns:
            vision_features: (batch_size, num_images, hidden_size)
        """
        batch_size, num_images = images.shape[:2]
        
        # Process 2D images
        images_flat = images.view(-1, *images.shape[2:])
        image_features = self.image_encoder(images_flat)
        
        # Global average pooling for image features
        if len(image_features.shape) == 3:  # (batch*num_images, seq_len, hidden)
            image_features = image_features.mean(dim=1)  # (batch*num_images, hidden)
        elif len(image_features.shape) == 4:  # (batch*num_images, hidden, h, w)
            image_features = image_features.mean(dim=[2, 3])  # (batch*num_images, hidden)
        
        # Process 3D point clouds if available
        if point_clouds is not None:
            point_features = self.point_encoder(point_clouds)  # (batch_size, point_dim)
            # Repeat point features for each image
            point_features = point_features.unsqueeze(1).repeat(1, num_images, 1)
            point_features = point_features.view(-1, point_features.shape[-1])
            
            # Fuse 2D and 3D features
            combined_features = torch.cat([image_features, point_features], dim=-1)
            vision_features = self.feature_fusion(combined_features)
        else:
            # Use only 2D features
            vision_features = image_features
            
        # Reshape back to (batch_size, num_images, hidden_size)
        vision_features = vision_features.view(batch_size, num_images, -1)
        
        return vision_features


class LanguageEncoder(nn.Module):
    """
    Language encoder based on LLaMA-2
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-hf",
        pretrained_path: Optional[str] = None,
        freeze_lm: bool = False,
        hidden_size: int = 4096,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.hidden_size = hidden_size
        
        # Load language model
        self.language_model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        
        self.language_hidden_size = self.language_model.config.hidden_size
        
        # Projection layer to match hidden size
        if self.language_hidden_size != hidden_size:
            self.language_proj = nn.Linear(self.language_hidden_size, hidden_size)
        else:
            self.language_proj = nn.Identity()
            
        # Freeze language model if specified
        if freeze_lm:
            for param in self.language_model.parameters():
                param.requires_grad = False
                
        # Load pretrained weights if provided
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)
    
    def load_pretrained_weights(self, pretrained_path: str):
        """Load pretrained language model weights"""
        try:
            checkpoint = torch.load(pretrained_path, map_location="cpu")
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            self.language_model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded pretrained language weights from {pretrained_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights from {pretrained_path}: {e}")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through language encoder
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            
        Returns:
            language_features: (batch_size, seq_len, hidden_size)
        """
        # Get language model outputs
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        
        # Extract hidden states
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden)
        
        # Project to target hidden size
        hidden_states = self.language_proj(hidden_states)
        
        return hidden_states


class MultimodalFusion(nn.Module):
    """
    Multimodal fusion module using cross-attention
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        num_heads: int = 32,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        
        # Final projection
        self.final_proj = nn.Linear(hidden_size, hidden_size)
        
    def forward(
        self,
        vision_features: torch.Tensor,
        language_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        language_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through multimodal fusion
        
        Args:
            vision_features: (batch_size, num_images, hidden_size)
            language_features: (batch_size, seq_len, hidden_size)
            vision_mask: (batch_size, num_images) - 1 for valid images, 0 for padding
            language_mask: (batch_size, seq_len) - 1 for valid tokens, 0 for padding
            
        Returns:
            fused_features: (batch_size, seq_len, hidden_size)
        """
        # Use language features as query, vision features as key/value
        query = language_features
        key_value = vision_features
        
        # Create attention mask for vision features
        if vision_mask is not None:
            # Convert to attention mask format (True for valid positions)
            vision_attention_mask = vision_mask.bool()
        else:
            vision_attention_mask = None
            
        # Apply cross-attention layers
        for i, (cross_attn, layer_norm, ffn) in enumerate(
            zip(self.cross_attention_layers, self.layer_norms, self.ffns)
        ):
            # Cross-attention
            attn_output, _ = cross_attn(
                query=query,
                key=key_value,
                value=key_value,
                key_padding_mask=~vision_attention_mask if vision_attention_mask is not None else None,
            )
            
            # Residual connection and layer norm
            query = layer_norm(query + attn_output)
            
            # Feed-forward network
            ffn_output = ffn(query)
            query = query + ffn_output
            
        # Final projection
        fused_features = self.final_proj(query)
        
        return fused_features


class NavR1Backbone(nn.Module):
    """
    Nav-R1 backbone model combining vision, language, and multimodal fusion
    """
    
    def __init__(
        self,
        vision_config: Dict[str, Any],
        language_config: Dict[str, Any],
        fusion_config: Dict[str, Any],
    ):
        super().__init__()
        
        # Initialize encoders
        self.vision_encoder = VisionEncoder(**vision_config)
        self.language_encoder = LanguageEncoder(**language_config)
        self.multimodal_fusion = MultimodalFusion(**fusion_config)
        
        # Hidden size
        self.hidden_size = fusion_config["hidden_size"]
        
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_mask: Optional[torch.Tensor] = None,
        point_clouds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through Nav-R1 backbone
        
        Args:
            images: (batch_size, num_images, 3, height, width)
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            image_mask: (batch_size, num_images) - 1 for valid images, 0 for padding
            point_clouds: (batch_size, num_points, 3) - optional 3D point clouds
            
        Returns:
            fused_features: (batch_size, seq_len, hidden_size)
        """
        # Encode vision (with optional 3D point clouds)
        vision_features = self.vision_encoder(images, point_clouds)  # (batch_size, num_images, hidden_size)
        
        # Encode language
        language_features = self.language_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )  # (batch_size, seq_len, hidden_size)
        
        # Multimodal fusion
        fused_features = self.multimodal_fusion(
            vision_features=vision_features,
            language_features=language_features,
            vision_mask=image_mask,
            language_mask=attention_mask,
        )  # (batch_size, seq_len, hidden_size)
        
        return fused_features
    
    def get_vision_features(self, images: torch.Tensor) -> torch.Tensor:
        """Get vision features only"""
        return self.vision_encoder(images)
    
    def get_language_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Get language features only"""
        return self.language_encoder(input_ids, attention_mask)
