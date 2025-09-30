"""
Embodied Tasks Datasets for Nav-R1
Includes: Dialogue, Reasoning, Planning, Navigation (VLN & ObjectNav)
"""

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image
import cv2
from transformers import AutoTokenizer


class EmbodiedDialogueDataset(Dataset):
    """
    Embodied Dialogue Dataset for conversational navigation
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_sequence_length: int = 1024,
        max_images: int = 8,
        image_size: int = 224,
        tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
    ):
        self.data_path = data_path
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.max_images = max_images
        self.image_size = image_size
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load embodied dialogue data"""
        file_path = os.path.join(self.data_path, f"embodied_dialogue_{self.split}.jsonl")
        
        if not os.path.exists(file_path):
            # Create dummy data for demonstration
            return self._create_dummy_dialogue_data()
            
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
                    
        return data
    
    def _create_dummy_dialogue_data(self) -> List[Dict[str, Any]]:
        """Create dummy dialogue data for demonstration"""
        dummy_data = []
        for i in range(100):
            dummy_data.append({
                "dialogue_id": f"dialogue_{i:04d}",
                "scene_id": f"scene_{i % 10:03d}",
                "conversation": [
                    {"role": "user", "content": "Can you help me find the kitchen?"},
                    {"role": "assistant", "content": "I'll help you navigate to the kitchen. Let me look around first."},
                    {"role": "user", "content": "What do you see?"},
                    {"role": "assistant", "content": "I can see a hallway with doors on both sides. Let me check which one leads to the kitchen."}
                ],
                "images": [f"image_{j}.jpg" for j in range(8)],
                "actions": ["LOOK_AROUND", "MOVE_FORWARD", "TURN_LEFT", "STOP"],
                "target_location": "kitchen"
            })
        return dummy_data
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        if not os.path.exists(image_path):
            # Return dummy image
            return torch.randn(3, self.image_size, self.image_size)
            
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.image_size, self.image_size))
            image = torch.from_numpy(np.array(image)).float() / 255.0
            image = image.permute(2, 0, 1)  # HWC -> CHW
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.randn(3, self.image_size, self.image_size)
    
    def _process_images(self, image_paths: List[str]) -> torch.Tensor:
        """Process dialogue images"""
        images = []
        
        for img_path in image_paths[:self.max_images]:
            image = self._load_image(img_path)
            images.append(image)
            
        # Pad with dummy images if needed
        while len(images) < self.max_images:
            images.append(torch.randn(3, self.image_size, self.image_size))
            
        return torch.stack(images)
    
    def _process_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Process conversation into text"""
        text_parts = []
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            text_parts.append(f"{role}: {content}")
        return "\n".join(text_parts)
    
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
        """Get a single dialogue sample"""
        sample = self.data[idx]
        
        # Process conversation
        conversation_text = self._process_conversation(sample["conversation"])
        
        # Process images
        images = self._process_images(sample["images"])
        
        # Tokenize conversation
        text_encoding = self._tokenize_text(conversation_text)
        
        return {
            "dialogue_id": sample["dialogue_id"],
            "scene_id": sample["scene_id"],
            "conversation": conversation_text,
            "images": images,
            "actions": sample["actions"],
            "target_location": sample["target_location"],
            "input_ids": text_encoding["input_ids"],
            "attention_mask": text_encoding["attention_mask"],
            "image_mask": torch.tensor([1.0] * len(sample["images"]) + [0.0] * (self.max_images - len(sample["images"])))
        }


class EmbodiedReasoningDataset(Dataset):
    """
    Embodied Reasoning Dataset for spatial reasoning tasks
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_sequence_length: int = 1024,
        max_images: int = 8,
        image_size: int = 224,
        tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
    ):
        self.data_path = data_path
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.max_images = max_images
        self.image_size = image_size
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load embodied reasoning data"""
        file_path = os.path.join(self.data_path, f"embodied_reasoning_{self.split}.jsonl")
        
        if not os.path.exists(file_path):
            # Create dummy data for demonstration
            return self._create_dummy_reasoning_data()
            
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
                    
        return data
    
    def _create_dummy_reasoning_data(self) -> List[Dict[str, Any]]:
        """Create dummy reasoning data for demonstration"""
        dummy_data = []
        reasoning_tasks = [
            "spatial_relationships",
            "object_counting",
            "path_planning",
            "obstacle_avoidance",
            "room_identification"
        ]
        
        for i in range(200):
            task_type = reasoning_tasks[i % len(reasoning_tasks)]
            dummy_data.append({
                "reasoning_id": f"reasoning_{i:04d}",
                "scene_id": f"scene_{i % 20:03d}",
                "task_type": task_type,
                "question": f"What is the relationship between the {['chair', 'table', 'sofa', 'bed'][i % 4]} and the {['window', 'door', 'wall', 'floor'][i % 4]}?",
                "reasoning_steps": [
                    "First, I need to identify the objects in the scene",
                    "Then, I'll analyze their spatial positions",
                    "Finally, I'll determine their relationship"
                ],
                "answer": f"The {['chair', 'table', 'sofa', 'bed'][i % 4]} is {['near', 'far from', 'behind', 'in front of'][i % 4]} the {['window', 'door', 'wall', 'floor'][i % 4]}",
                "images": [f"image_{j}.jpg" for j in range(8)],
                "difficulty": ["easy", "medium", "hard"][i % 3]
            })
        return dummy_data
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        if not os.path.exists(image_path):
            return torch.randn(3, self.image_size, self.image_size)
            
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.image_size, self.image_size))
            image = torch.from_numpy(np.array(image)).float() / 255.0
            image = image.permute(2, 0, 1)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.randn(3, self.image_size, self.image_size)
    
    def _process_images(self, image_paths: List[str]) -> torch.Tensor:
        """Process reasoning images"""
        images = []
        
        for img_path in image_paths[:self.max_images]:
            image = self._load_image(img_path)
            images.append(image)
            
        while len(images) < self.max_images:
            images.append(torch.randn(3, self.image_size, self.image_size))
            
        return torch.stack(images)
    
    def _process_reasoning_task(self, sample: Dict[str, Any]) -> str:
        """Process reasoning task into text"""
        text_parts = []
        text_parts.append(f"Task: {sample['task_type']}")
        text_parts.append(f"Question: {sample['question']}")
        text_parts.append("Reasoning Steps:")
        for i, step in enumerate(sample['reasoning_steps']):
            text_parts.append(f"{i+1}. {step}")
        text_parts.append(f"Answer: {sample['answer']}")
        return "\n".join(text_parts)
    
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
        """Get a single reasoning sample"""
        sample = self.data[idx]
        
        # Process reasoning task
        reasoning_text = self._process_reasoning_task(sample)
        
        # Process images
        images = self._process_images(sample["images"])
        
        # Tokenize reasoning text
        text_encoding = self._tokenize_text(reasoning_text)
        
        return {
            "reasoning_id": sample["reasoning_id"],
            "scene_id": sample["scene_id"],
            "task_type": sample["task_type"],
            "question": sample["question"],
            "reasoning_steps": sample["reasoning_steps"],
            "answer": sample["answer"],
            "images": images,
            "difficulty": sample["difficulty"],
            "input_ids": text_encoding["input_ids"],
            "attention_mask": text_encoding["attention_mask"],
            "image_mask": torch.tensor([1.0] * len(sample["images"]) + [0.0] * (self.max_images - len(sample["images"])))
        }


class EmbodiedPlanningDataset(Dataset):
    """
    Embodied Planning Dataset for task planning and execution
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_sequence_length: int = 1024,
        max_images: int = 8,
        image_size: int = 224,
        tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
    ):
        self.data_path = data_path
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.max_images = max_images
        self.image_size = image_size
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load embodied planning data"""
        file_path = os.path.join(self.data_path, f"embodied_planning_{self.split}.jsonl")
        
        if not os.path.exists(file_path):
            # Create dummy data for demonstration
            return self._create_dummy_planning_data()
            
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
                    
        return data
    
    def _create_dummy_planning_data(self) -> List[Dict[str, Any]]:
        """Create dummy planning data for demonstration"""
        dummy_data = []
        planning_tasks = [
            "multi_object_navigation",
            "sequential_task_execution",
            "resource_gathering",
            "room_cleaning",
            "object_manipulation"
        ]
        
        for i in range(150):
            task_type = planning_tasks[i % len(planning_tasks)]
            dummy_data.append({
                "planning_id": f"planning_{i:04d}",
                "scene_id": f"scene_{i % 15:03d}",
                "task_type": task_type,
                "goal": f"Find and collect {['3 red objects', 'all books', 'kitchen utensils', 'cleaning supplies'][i % 4]}",
                "constraints": [
                    "Avoid obstacles",
                    "Complete within 5 minutes",
                    "Don't break anything"
                ],
                "plan": [
                    "1. Survey the current room",
                    "2. Identify target objects",
                    "3. Plan optimal path",
                    "4. Execute navigation",
                    "5. Collect objects",
                    "6. Return to starting position"
                ],
                "subgoals": [
                    "Navigate to kitchen",
                    "Find red objects",
                    "Collect items",
                    "Return to living room"
                ],
                "images": [f"image_{j}.jpg" for j in range(8)],
                "success_criteria": "All target objects collected and returned to starting position"
            })
        return dummy_data
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        if not os.path.exists(image_path):
            return torch.randn(3, self.image_size, self.image_size)
            
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.image_size, self.image_size))
            image = torch.from_numpy(np.array(image)).float() / 255.0
            image = image.permute(2, 0, 1)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.randn(3, self.image_size, self.image_size)
    
    def _process_images(self, image_paths: List[str]) -> torch.Tensor:
        """Process planning images"""
        images = []
        
        for img_path in image_paths[:self.max_images]:
            image = self._load_image(img_path)
            images.append(image)
            
        while len(images) < self.max_images:
            images.append(torch.randn(3, self.image_size, self.image_size))
            
        return torch.stack(images)
    
    def _process_planning_task(self, sample: Dict[str, Any]) -> str:
        """Process planning task into text"""
        text_parts = []
        text_parts.append(f"Task: {sample['task_type']}")
        text_parts.append(f"Goal: {sample['goal']}")
        text_parts.append("Constraints:")
        for constraint in sample['constraints']:
            text_parts.append(f"- {constraint}")
        text_parts.append("Plan:")
        for step in sample['plan']:
            text_parts.append(step)
        text_parts.append("Subgoals:")
        for i, subgoal in enumerate(sample['subgoals']):
            text_parts.append(f"{i+1}. {subgoal}")
        text_parts.append(f"Success Criteria: {sample['success_criteria']}")
        return "\n".join(text_parts)
    
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
        """Get a single planning sample"""
        sample = self.data[idx]
        
        # Process planning task
        planning_text = self._process_planning_task(sample)
        
        # Process images
        images = self._process_images(sample["images"])
        
        # Tokenize planning text
        text_encoding = self._tokenize_text(planning_text)
        
        return {
            "planning_id": sample["planning_id"],
            "scene_id": sample["scene_id"],
            "task_type": sample["task_type"],
            "goal": sample["goal"],
            "constraints": sample["constraints"],
            "plan": sample["plan"],
            "subgoals": sample["subgoals"],
            "images": images,
            "success_criteria": sample["success_criteria"],
            "input_ids": text_encoding["input_ids"],
            "attention_mask": text_encoding["attention_mask"],
            "image_mask": torch.tensor([1.0] * len(sample["images"]) + [0.0] * (self.max_images - len(sample["images"])))
        }


class VLNDataset(Dataset):
    """
    Vision-and-Language Navigation Dataset
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_sequence_length: int = 512,
        max_images: int = 8,
        image_size: int = 224,
        tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
    ):
        self.data_path = data_path
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.max_images = max_images
        self.image_size = image_size
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load VLN data"""
        file_path = os.path.join(self.data_path, f"vln_{self.split}.json")
        
        if not os.path.exists(file_path):
            # Create dummy data for demonstration
            return self._create_dummy_vln_data()
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return data
    
    def _create_dummy_vln_data(self) -> List[Dict[str, Any]]:
        """Create dummy VLN data for demonstration"""
        dummy_data = []
        instructions = [
            "Go to the kitchen and find the red chair",
            "Navigate to the living room and stop near the sofa",
            "Find the bathroom and look for a mirror",
            "Go to the bedroom and find the bed",
            "Navigate to the dining room and find the table"
        ]
        
        for i in range(300):
            dummy_data.append({
                "episode_id": f"vln_episode_{i:04d}",
                "scene_id": f"scene_{i % 25:03d}",
                "instruction": instructions[i % len(instructions)],
                "path": [
                    {"position": [0.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
                    {"position": [1.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
                    {"position": [2.0, 0.0, 1.0], "rotation": [0.0, 0.0, 0.0, 1.0]},
                    {"position": [3.0, 0.0, 2.0], "rotation": [0.0, 0.0, 0.0, 1.0]}
                ],
                "goals": [
                    {"position": [3.0, 0.0, 2.0], "radius": 1.0}
                ],
                "start_position": [0.0, 0.0, 0.0],
                "start_rotation": [0.0, 0.0, 0.0, 1.0],
                "images": [f"image_{j}.jpg" for j in range(8)],
                "trajectory_length": 4,
                "success": i % 3 == 0  # 33% success rate
            })
        return dummy_data
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        if not os.path.exists(image_path):
            return torch.randn(3, self.image_size, self.image_size)
            
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.image_size, self.image_size))
            image = torch.from_numpy(np.array(image)).float() / 255.0
            image = image.permute(2, 0, 1)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.randn(3, self.image_size, self.image_size)
    
    def _process_images(self, image_paths: List[str]) -> torch.Tensor:
        """Process VLN images"""
        images = []
        
        for img_path in image_paths[:self.max_images]:
            image = self._load_image(img_path)
            images.append(image)
            
        while len(images) < self.max_images:
            images.append(torch.randn(3, self.image_size, self.image_size))
            
        return torch.stack(images)
    
    def _process_instruction(self, instruction: str) -> str:
        """Process VLN instruction"""
        return f"Navigation Instruction: {instruction}"
    
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
        """Get a single VLN sample"""
        sample = self.data[idx]
        
        # Process instruction
        instruction_text = self._process_instruction(sample["instruction"])
        
        # Process images
        images = self._process_images(sample["images"])
        
        # Tokenize instruction
        text_encoding = self._tokenize_text(instruction_text)
        
        return {
            "episode_id": sample["episode_id"],
            "scene_id": sample["scene_id"],
            "instruction": sample["instruction"],
            "path": sample["path"],
            "goals": sample["goals"],
            "start_position": sample["start_position"],
            "start_rotation": sample["start_rotation"],
            "images": images,
            "trajectory_length": sample["trajectory_length"],
            "success": sample["success"],
            "input_ids": text_encoding["input_ids"],
            "attention_mask": text_encoding["attention_mask"],
            "image_mask": torch.tensor([1.0] * len(sample["images"]) + [0.0] * (self.max_images - len(sample["images"])))
        }


class ObjectNavDataset(Dataset):
    """
    Object Navigation Dataset
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = "train",
        max_sequence_length: int = 512,
        max_images: int = 8,
        image_size: int = 224,
        tokenizer_name: str = "meta-llama/Llama-2-7b-hf",
    ):
        self.data_path = data_path
        self.split = split
        self.max_sequence_length = max_sequence_length
        self.max_images = max_images
        self.image_size = image_size
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load data
        self.data = self._load_data()
        
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load ObjectNav data"""
        file_path = os.path.join(self.data_path, f"objectnav_{self.split}.json")
        
        if not os.path.exists(file_path):
            # Create dummy data for demonstration
            return self._create_dummy_objectnav_data()
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return data
    
    def _create_dummy_objectnav_data(self) -> List[Dict[str, Any]]:
        """Create dummy ObjectNav data for demonstration"""
        dummy_data = []
        object_categories = [
            "chair", "table", "sofa", "bed", "toilet", "tv", "laptop", "book",
            "bottle", "cup", "bowl", "banana", "apple", "orange", "broccoli",
            "carrot", "hot dog", "pizza", "donut", "cake"
        ]
        
        for i in range(400):
            object_category = object_categories[i % len(object_categories)]
            dummy_data.append({
                "episode_id": f"objectnav_episode_{i:04d}",
                "scene_id": f"scene_{i % 30:03d}",
                "object_category": object_category,
                "object_id": f"{object_category}_{i}",
                "start_position": [0.0, 0.0, 0.0],
                "start_rotation": [0.0, 0.0, 0.0, 1.0],
                "goals": [
                    {"position": [2.0, 0.0, 1.0], "radius": 0.5, "object_id": f"{object_category}_{i}"}
                ],
                "images": [f"image_{j}.jpg" for j in range(8)],
                "trajectory_length": 5,
                "success": i % 4 == 0,  # 25% success rate
                "spl": 0.8 if i % 4 == 0 else 0.0
            })
        return dummy_data
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image"""
        if not os.path.exists(image_path):
            return torch.randn(3, self.image_size, self.image_size)
            
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize((self.image_size, self.image_size))
            image = torch.from_numpy(np.array(image)).float() / 255.0
            image = image.permute(2, 0, 1)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return torch.randn(3, self.image_size, self.image_size)
    
    def _process_images(self, image_paths: List[str]) -> torch.Tensor:
        """Process ObjectNav images"""
        images = []
        
        for img_path in image_paths[:self.max_images]:
            image = self._load_image(img_path)
            images.append(image)
            
        while len(images) < self.max_images:
            images.append(torch.randn(3, self.image_size, self.image_size))
            
        return torch.stack(images)
    
    def _process_object_task(self, object_category: str) -> str:
        """Process object navigation task"""
        return f"Find and navigate to the {object_category}"
    
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
        """Get a single ObjectNav sample"""
        sample = self.data[idx]
        
        # Process object task
        task_text = self._process_object_task(sample["object_category"])
        
        # Process images
        images = self._process_images(sample["images"])
        
        # Tokenize task text
        text_encoding = self._tokenize_text(task_text)
        
        return {
            "episode_id": sample["episode_id"],
            "scene_id": sample["scene_id"],
            "object_category": sample["object_category"],
            "object_id": sample["object_id"],
            "start_position": sample["start_position"],
            "start_rotation": sample["start_rotation"],
            "goals": sample["goals"],
            "images": images,
            "trajectory_length": sample["trajectory_length"],
            "success": sample["success"],
            "spl": sample["spl"],
            "input_ids": text_encoding["input_ids"],
            "attention_mask": text_encoding["attention_mask"],
            "image_mask": torch.tensor([1.0] * len(sample["images"]) + [0.0] * (self.max_images - len(sample["images"])))
        }


def create_embodied_dataset(
    task_type: str,
    data_path: str,
    split: str = "train",
    **kwargs
) -> Dataset:
    """Factory function to create embodied task dataset"""
    if task_type == "dialogue":
        return EmbodiedDialogueDataset(data_path, split, **kwargs)
    elif task_type == "reasoning":
        return EmbodiedReasoningDataset(data_path, split, **kwargs)
    elif task_type == "planning":
        return EmbodiedPlanningDataset(data_path, split, **kwargs)
    elif task_type == "vln":
        return VLNDataset(data_path, split, **kwargs)
    elif task_type == "objectnav":
        return ObjectNavDataset(data_path, split, **kwargs)
    else:
        raise ValueError(f"Unknown embodied task type: {task_type}")


def create_embodied_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Create DataLoader for embodied task dataset"""
    def collate_fn(batch):
        """Custom collate function for embodied data"""
        collated = {}
        
        # Stack tensors
        for key in ["images", "input_ids", "attention_mask", "image_mask"]:
            if key in batch[0]:
                collated[key] = torch.stack([item[key] for item in batch])
                
        # Keep lists as lists
        for key in batch[0].keys():
            if key not in ["images", "input_ids", "attention_mask", "image_mask"]:
                if isinstance(batch[0][key], (list, str, int, float, bool)):
                    collated[key] = [item[key] for item in batch]
                elif isinstance(batch[0][key], torch.Tensor):
                    collated[key] = torch.stack([item[key] for item in batch])
                    
        return collated
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn
    )
