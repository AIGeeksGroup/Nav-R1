"""
Task-specific data loaders for different embodied navigation tasks
"""

import json
import os
from typing import Dict, List, Any, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from .nav_cot import NavCoTDataset


class VLNTaskLoader(Dataset):
    """Vision-and-Language Navigation task loader"""
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_episode_length: int = 500,
        **kwargs
    ):
        self.data_path = data_path
        self.split = split
        self.max_episode_length = max_episode_length
        self.data = self._load_episodes()
        
    def _load_episodes(self) -> List[Dict[str, Any]]:
        """Load VLN episodes"""
        file_path = os.path.join(self.data_path, f"{self.split}.json.gz")
        
        if not os.path.exists(file_path):
            # Try uncompressed version
            file_path = os.path.join(self.data_path, f"{self.split}.json")
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Episode file not found: {file_path}")
            
        import gzip
        if file_path.endswith('.gz'):
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single episode"""
        episode = self.data[idx]
        
        return {
            "episode_id": episode.get("episode_id", idx),
            "instruction": episode.get("instruction", {}).get("instruction_text", ""),
            "path": episode.get("reference_path", []),
            "goals": episode.get("goals", []),
            "scene_id": episode.get("scene_id", ""),
            "start_position": episode.get("start_position", []),
            "start_rotation": episode.get("start_rotation", []),
            "info": episode.get("info", {})
        }


class ObjectNavTaskLoader(Dataset):
    """Object Navigation task loader"""
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        **kwargs
    ):
        self.data_path = data_path
        self.split = split
        self.data = self._load_episodes()
        
    def _load_episodes(self) -> List[Dict[str, Any]]:
        """Load ObjectNav episodes"""
        file_path = os.path.join(self.data_path, f"{self.split}.json")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Episode file not found: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single episode"""
        episode = self.data[idx]
        
        return {
            "episode_id": episode.get("episode_id", idx),
            "object_category": episode.get("object_category", ""),
            "start_position": episode.get("start_position", []),
            "start_rotation": episode.get("start_rotation", []),
            "goals": episode.get("goals", []),
            "scene_id": episode.get("scene_id", ""),
            "info": episode.get("info", {})
        }


class TaskDataLoader:
    """Unified task data loader"""
    
    def __init__(
        self,
        task_type: str,
        data_path: str,
        split: str = "train",
        batch_size: int = 8,
        **kwargs
    ):
        self.task_type = task_type
        self.data_path = data_path
        self.split = split
        self.batch_size = batch_size
        
        # Create appropriate dataset
        if task_type == "vln":
            self.dataset = VLNTaskLoader(data_path, split, **kwargs)
        elif task_type == "objectnav":
            self.dataset = ObjectNavTaskLoader(data_path, split, **kwargs)
        elif task_type == "nav_cot":
            self.dataset = NavCoTDataset(data_path, split, **kwargs)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def get_dataloader(self, shuffle: bool = True, num_workers: int = 4) -> DataLoader:
        """Get PyTorch DataLoader"""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function"""
        if self.task_type == "nav_cot":
            # Use NavCoT collate function
            from .nav_cot import NavCoTDataLoader
            nav_cot_loader = NavCoTDataLoader(self.dataset, self.batch_size)
            return nav_cot_loader._collate_fn(batch)
        else:
            # Simple collation for other tasks
            collated = {}
            for key in batch[0].keys():
                if isinstance(batch[0][key], (list, str)):
                    collated[key] = [item[key] for item in batch]
                else:
                    collated[key] = torch.stack([item[key] for item in batch])
            return collated


def create_task_dataloader(
    task_type: str,
    data_path: str,
    split: str = "train",
    **kwargs
) -> TaskDataLoader:
    """Factory function to create task-specific dataloader"""
    return TaskDataLoader(task_type, data_path, split, **kwargs)
