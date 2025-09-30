"""
3D Scene Datasets for Nav-R1 (based on 3D-R1)
"""

import json
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import cv2
from transformers import AutoTokenizer


class ScanReferDataset(Dataset):
    """
    ScanRefer dataset for 3D visual grounding
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_sequence_length: int = 512,
        max_points: int = 4096,
        image_size: int = 224,
        tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
    ):
        self.data_path = data_path
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.max_points = max_points
        self.image_size = image_size
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load ScanRefer data"""
        file_path = os.path.join(self.data_path, f"ScanRefer_filtered_{self.split}.json")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        return data
    
    def _load_point_cloud(self, scene_id: str) -> torch.Tensor:
        """Load 3D point cloud"""
        # This is a simplified implementation
        # In practice, you would load from .ply files
        point_cloud_path = os.path.join(self.data_path, "scans", scene_id, f"{scene_id}.ply")
        
        if os.path.exists(point_cloud_path):
            # Load point cloud (simplified)
            points = np.random.randn(self.max_points, 3).astype(np.float32)
        else:
            # Generate dummy point cloud
            points = np.random.randn(self.max_points, 3).astype(np.float32)
            
        return torch.from_numpy(points)
    
    def _load_images(self, scene_id: str) -> torch.Tensor:
        """Load RGB images"""
        # This is a simplified implementation
        # In practice, you would load from actual image files
        images = []
        for i in range(8):  # Assume 8 views
            # Generate dummy image
            image = torch.randn(3, self.image_size, self.image_size)
            images.append(image)
            
        return torch.stack(images)
    
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
        
        # Extract information
        scene_id = sample["scene_id"]
        object_id = sample["object_id"]
        description = sample["description"]
        
        # Load point cloud
        point_cloud = self._load_point_cloud(scene_id)
        
        # Load images
        images = self._load_images(scene_id)
        
        # Tokenize description
        text_encoding = self._tokenize_text(description)
        
        return {
            "scene_id": scene_id,
            "object_id": object_id,
            "description": description,
            "point_cloud": point_cloud,
            "images": images,
            "input_ids": text_encoding["input_ids"],
            "attention_mask": text_encoding["attention_mask"],
            "image_mask": torch.ones(8),  # All images are valid
        }


class ScanQADataset(Dataset):
    """
    ScanQA dataset for 3D question answering
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_sequence_length: int = 512,
        max_points: int = 4096,
        image_size: int = 224,
        tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
    ):
        self.data_path = data_path
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.max_points = max_points
        self.image_size = image_size
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load ScanQA data"""
        file_path = os.path.join(self.data_path, f"ScanQA_v1.0_{self.split}.json")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        return data
    
    def _load_point_cloud(self, scene_id: str) -> torch.Tensor:
        """Load 3D point cloud"""
        # Simplified implementation
        points = np.random.randn(self.max_points, 3).astype(np.float32)
        return torch.from_numpy(points)
    
    def _load_images(self, scene_id: str) -> torch.Tensor:
        """Load RGB images"""
        images = []
        for i in range(8):
            image = torch.randn(3, self.image_size, self.image_size)
            images.append(image)
        return torch.stack(images)
    
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
        
        # Extract information
        scene_id = sample["scene_id"]
        question = sample["question"]
        answers = sample["answers"]
        
        # Load point cloud
        point_cloud = self._load_point_cloud(scene_id)
        
        # Load images
        images = self._load_images(scene_id)
        
        # Tokenize question
        text_encoding = self._tokenize_text(question)
        
        return {
            "scene_id": scene_id,
            "question": question,
            "answers": answers,
            "point_cloud": point_cloud,
            "images": images,
            "input_ids": text_encoding["input_ids"],
            "attention_mask": text_encoding["attention_mask"],
            "image_mask": torch.ones(8),
        }


class Nr3DDataset(Dataset):
    """
    Nr3D dataset for 3D visual grounding
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_sequence_length: int = 512,
        max_points: int = 4096,
        image_size: int = 224,
        tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
    ):
        self.data_path = data_path
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.max_points = max_points
        self.image_size = image_size
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load Nr3D data"""
        file_path = os.path.join(self.data_path, f"nr3d_{self.split}.json")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        return data
    
    def _load_point_cloud(self, scene_id: str) -> torch.Tensor:
        """Load 3D point cloud"""
        points = np.random.randn(self.max_points, 3).astype(np.float32)
        return torch.from_numpy(points)
    
    def _load_images(self, scene_id: str) -> torch.Tensor:
        """Load RGB images"""
        images = []
        for i in range(8):
            image = torch.randn(3, self.image_size, self.image_size)
            images.append(image)
        return torch.stack(images)
    
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
        
        # Extract information
        scene_id = sample["scan_id"]
        object_id = sample["target_id"]
        utterance = sample["utterance"]
        
        # Load point cloud
        point_cloud = self._load_point_cloud(scene_id)
        
        # Load images
        images = self._load_images(scene_id)
        
        # Tokenize utterance
        text_encoding = self._tokenize_text(utterance)
        
        return {
            "scene_id": scene_id,
            "object_id": object_id,
            "utterance": utterance,
            "point_cloud": point_cloud,
            "images": images,
            "input_ids": text_encoding["input_ids"],
            "attention_mask": text_encoding["attention_mask"],
            "image_mask": torch.ones(8),
        }


class Scene30KDataset(Dataset):
    """
    Scene-30K synthetic dataset for 3D scene understanding
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_sequence_length: int = 1024,
        max_points: int = 8192,
        image_size: int = 224,
        tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
    ):
        self.data_path = data_path
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.max_points = max_points
        self.image_size = image_size
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load Scene-30K data"""
        file_path = os.path.join(self.data_path, f"{self.split}.jsonl")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
                    
        return data
    
    def _load_point_cloud(self, scene_id: str) -> torch.Tensor:
        """Load 3D point cloud"""
        points = np.random.randn(self.max_points, 3).astype(np.float32)
        return torch.from_numpy(points)
    
    def _load_images(self, scene_id: str) -> torch.Tensor:
        """Load RGB images"""
        images = []
        for i in range(8):
            image = torch.randn(3, self.image_size, self.image_size)
            images.append(image)
        return torch.stack(images)
    
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
        
        # Extract information
        scene_id = sample.get("scene_id", f"scene_{idx}")
        instruction = sample.get("instruction", "")
        reasoning = sample.get("reasoning", "")
        
        # Load point cloud
        point_cloud = self._load_point_cloud(scene_id)
        
        # Load images
        images = self._load_images(scene_id)
        
        # Create full text
        full_text = f"Instruction: {instruction}\nReasoning: {reasoning}"
        text_encoding = self._tokenize_text(full_text)
        
        return {
            "scene_id": scene_id,
            "instruction": instruction,
            "reasoning": reasoning,
            "point_cloud": point_cloud,
            "images": images,
            "input_ids": text_encoding["input_ids"],
            "attention_mask": text_encoding["attention_mask"],
            "image_mask": torch.ones(8),
        }


def create_3d_dataset(
    dataset_name: str,
    data_path: str,
    split: str = "train",
    **kwargs
) -> Dataset:
    """Factory function to create 3D dataset"""
    if dataset_name == "scanrefer":
        return ScanReferDataset(data_path, split, **kwargs)
    elif dataset_name == "scanqa":
        return ScanQADataset(data_path, split, **kwargs)
    elif dataset_name == "nr3d":
        return Nr3DDataset(data_path, split, **kwargs)
    elif dataset_name == "scene30k":
        return Scene30KDataset(data_path, split, **kwargs)
    else:
        raise ValueError(f"Unknown 3D dataset: {dataset_name}")


def create_3d_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create DataLoader for 3D dataset"""
    def collate_fn(batch):
        """Custom collate function for 3D data"""
        collated = {}
        
        # Stack tensors
        for key in ["images", "point_cloud", "input_ids", "attention_mask", "image_mask"]:
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch])
                
        # Keep lists as lists
        for key in ["scene_id", "description", "question", "answers", "utterance", "instruction", "reasoning"]:
            if key in batch[0]:
                collated[key] = [item[key] for item in batch]
                
        return collated
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
