"""
Nav-CoT-110K Dataset Implementation
"""

import json
import os
from typing import Dict, List, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from transformers import AutoTokenizer
import cv2


class NavCoTDataset(Dataset):
    """
    Nav-CoT-110K Dataset for embodied reasoning and navigation
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_sequence_length: int = 2048,
        max_images: int = 8,
        image_size: int = 224,
        tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
        transform=None,
    ):
        self.data_path = data_path
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.max_images = max_images
        self.image_size = image_size
        self.transform = transform
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSONL files"""
        file_path = os.path.join(self.data_path, f"{self.split}.jsonl")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
                    
        return data
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        if not os.path.exists(image_path):
            # Return zero tensor if image doesn't exist
            return torch.zeros(3, self.image_size, self.image_size)
            
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.image_size, self.image_size))
            
            if self.transform:
                image = self.transform(image)
            else:
                # Default transform: normalize to [0, 1] and convert to tensor
                image = torch.from_numpy(np.array(image)).float() / 255.0
                image = image.permute(2, 0, 1)  # HWC -> CHW
                
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.zeros(3, self.image_size, self.image_size)
    
    def _process_images(self, history_images: List[str]) -> torch.Tensor:
        """Process history images"""
        images = []
        
        # Limit number of images
        for img_path in history_images[:self.max_images]:
            image = self._load_image(img_path)
            images.append(image)
            
        # Pad with zero tensors if needed
        while len(images) < self.max_images:
            images.append(torch.zeros(3, self.image_size, self.image_size))
            
        return torch.stack(images)
    
    def _process_instruction(self, instruction: str) -> str:
        """Process instruction text"""
        return instruction.strip()
    
    def _process_cot_reasoning(self, cot_data: Dict[str, Any]) -> str:
        """Process Chain-of-Thought reasoning"""
        reasoning_parts = []
        
        # Add observation
        if "observation" in cot_data:
            reasoning_parts.append(f"Observation: {cot_data['observation']}")
            
        # Add reasoning steps
        if "reasoning_steps" in cot_data:
            for i, step in enumerate(cot_data["reasoning_steps"]):
                reasoning_parts.append(f"Step {i+1}: {step}")
                
        # Add conclusion
        if "conclusion" in cot_data:
            reasoning_parts.append(f"Conclusion: {cot_data['conclusion']}")
            
        return "\n".join(reasoning_parts)
    
    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text"""
        encoding = self.tokenizer(
            text,
            max_length=self.max_sequence_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single data sample"""
        sample = self.data[idx]
        
        # Process instruction
        instruction = self._process_instruction(sample["instruction"])
        
        # Process images
        history_images = sample.get("history_images", [])
        images = self._process_images(history_images)
        
        # Process action space
        action_space = sample.get("action_space", [])
        
        # Process target
        target = sample.get("target", "")
        
        # Process CoT reasoning if available
        cot_reasoning = ""
        if "cot_reasoning" in sample:
            cot_reasoning = self._process_cot_reasoning(sample["cot_reasoning"])
        
        # Create full text for tokenization
        full_text = f"Instruction: {instruction}\n"
        if cot_reasoning:
            full_text += f"Reasoning: {cot_reasoning}\n"
        full_text += f"Action Space: {', '.join(action_space)}\n"
        full_text += f"Target: {target}"
        
        # Tokenize text
        text_encoding = self._tokenize_text(full_text)
        
        return {
            "instruction": instruction,
            "images": images,
            "action_space": action_space,
            "target": target,
            "cot_reasoning": cot_reasoning,
            "input_ids": text_encoding["input_ids"],
            "attention_mask": text_encoding["attention_mask"],
            "image_mask": torch.tensor([1.0] * len(history_images) + [0.0] * (self.max_images - len(history_images)))
        }


class NavCoTDataLoader:
    """DataLoader wrapper for Nav-CoT dataset"""
    
    def __init__(
        self,
        dataset: NavCoTDataset,
        batch_size: int = 8,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    def get_dataloader(self) -> DataLoader:
        """Get PyTorch DataLoader"""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching"""
        collated = {}
        
        # Stack tensors
        for key in ["images", "input_ids", "attention_mask", "image_mask"]:
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch])
                
        # Keep lists as lists
        for key in ["instruction", "action_space", "target", "cot_reasoning"]:
            if key in batch[0]:
                collated[key] = [item[key] for item in batch]
                
        return collated


def create_nav_cot_dataset(
    data_path: str,
    split: str = "train",
    **kwargs
) -> NavCoTDataset:
    """Factory function to create Nav-CoT dataset"""
    return NavCoTDataset(data_path=data_path, split=split, **kwargs)


def create_nav_cot_dataloader(
    dataset: NavCoTDataset,
    batch_size: int = 8,
    **kwargs
) -> DataLoader:
    """Factory function to create Nav-CoT dataloader"""
    loader = NavCoTDataLoader(dataset=dataset, batch_size=batch_size, **kwargs)
    return loader.get_dataloader()
