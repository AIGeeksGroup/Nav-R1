"""
Habitat Stub Simulator for testing without Habitat installation
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple


class HabitatStubSimulator:
    """
    Stub simulator for when Habitat is not available
    Provides the same interface as HabitatSimulator but with dummy data
    """
    
    def __init__(
        self,
        config_path: str = "",
        scene_dataset_path: str = "",
        episode_dataset_path: str = "",
        max_episode_steps: int = 500,
        success_reward: float = 10.0,
        step_penalty: float = -0.01,
        collision_penalty: float = -1.0,
        device: str = "cuda"
    ):
        self.config_path = config_path
        self.scene_dataset_path = scene_dataset_path
        self.episode_dataset_path = episode_dataset_path
        self.max_episode_steps = max_episode_steps
        self.success_reward = success_reward
        self.step_penalty = step_penalty
        self.collision_penalty = collision_penalty
        self.device = device
        
        # Episode state
        self.current_episode_id = None
        self.episode_step_count = 0
        self.episode_reward = 0.0
        self.episode_success = False
        
        # Action mapping
        self.action_mapping = {
            "MOVE_FORWARD": 0,
            "TURN_LEFT": 1,
            "TURN_RIGHT": 2,
            "STOP": 3,
        }
        
        # Dummy episode data
        self.dummy_episodes = [
            {
                "episode_id": "episode_001",
                "instruction": "Go to the kitchen and find a red chair",
                "goals": [[1.0, 0.0, 2.0], [2.0, 0.0, 3.0]],
                "scene_id": "scene_001"
            },
            {
                "episode_id": "episode_002", 
                "instruction": "Navigate to the living room and stop near the sofa",
                "goals": [[3.0, 0.0, 1.0]],
                "scene_id": "scene_002"
            },
            {
                "episode_id": "episode_003",
                "instruction": "Find the bathroom and look for a mirror",
                "goals": [[0.0, 0.0, 4.0]],
                "scene_id": "scene_003"
            }
        ]
        
    def reset(self, episode_id: Optional[str] = None) -> Dict[str, Any]:
        """Reset environment to new episode"""
        if episode_id is not None:
            # Find specific episode
            episode_data = next(
                (ep for ep in self.dummy_episodes if ep["episode_id"] == episode_id),
                self.dummy_episodes[0]
            )
        else:
            # Random episode
            import random
            episode_data = random.choice(self.dummy_episodes)
            
        self.current_episode_id = episode_data["episode_id"]
        self.episode_step_count = 0
        self.episode_reward = 0.0
        self.episode_success = False
        
        # Generate dummy observations
        observations = self._generate_dummy_observations(episode_data)
        
        return observations
    
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute action and return next state"""
        if action not in self.action_mapping:
            raise ValueError(f"Unknown action: {action}")
            
        self.episode_step_count += 1
        
        # Calculate reward
        reward = self._calculate_reward(action)
        self.episode_reward += reward
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Generate new observations
        episode_data = next(
            (ep for ep in self.dummy_episodes if ep["episode_id"] == self.current_episode_id),
            self.dummy_episodes[0]
        )
        observations = self._generate_dummy_observations(episode_data)
        
        # Create info dict
        info = {
            "episode_id": self.current_episode_id,
            "step_count": self.episode_step_count,
            "episode_reward": self.episode_reward,
            "success": self.episode_success,
            "collision": self._check_collision(),
            "action": action
        }
        
        return observations, reward, done, info
    
    def _generate_dummy_observations(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate dummy observations"""
        # Generate random RGB image
        rgb = torch.rand(3, 224, 224, device=self.device)
        
        # Generate random depth image
        depth = torch.rand(1, 224, 224, device=self.device) * 5.0  # 0-5 meters
        
        # Generate random agent state
        position = torch.rand(3, device=self.device) * 10.0  # Random position
        rotation = torch.rand(4, device=self.device)  # Random quaternion
        rotation = rotation / torch.norm(rotation)  # Normalize quaternion
        
        # Generate random GPS and compass
        gps = torch.rand(2, device=self.device) * 10.0
        compass = torch.rand(1, device=self.device) * 2 * np.pi
        
        return {
            "rgb": rgb,
            "depth": depth,
            "agent_state": {
                "position": position,
                "rotation": rotation,
                "velocity": torch.zeros(3, device=self.device)
            },
            "gps": gps,
            "compass": compass,
            "episode_info": {
                "episode_id": episode_data["episode_id"],
                "instruction": episode_data["instruction"],
                "goals": episode_data["goals"],
                "scene_id": episode_data["scene_id"]
            }
        }
    
    def _calculate_reward(self, action: str) -> float:
        """Calculate reward for current step"""
        reward = self.step_penalty
        
        # Random success (for testing)
        if np.random.random() < 0.01:  # 1% chance of success
            reward += self.success_reward
            self.episode_success = True
            
        # Random collision (for testing)
        if np.random.random() < 0.05:  # 5% chance of collision
            reward += self.collision_penalty
            
        return reward
    
    def _check_collision(self) -> bool:
        """Check if agent collided (dummy implementation)"""
        return np.random.random() < 0.05  # 5% chance of collision
    
    def _is_episode_done(self) -> bool:
        """Check if episode is done"""
        # Check if max steps reached
        if self.episode_step_count >= self.max_episode_steps:
            return True
            
        # Check if episode is successful
        if self.episode_success:
            return True
            
        return False
    
    def get_episode_info(self) -> Dict[str, Any]:
        """Get current episode information"""
        if self.current_episode_id is None:
            return {}
            
        episode_data = next(
            (ep for ep in self.dummy_episodes if ep["episode_id"] == self.current_episode_id),
            self.dummy_episodes[0]
        )
        
        return {
            "episode_id": self.current_episode_id,
            "scene_id": episode_data["scene_id"],
            "step_count": self.episode_step_count,
            "episode_reward": self.episode_reward,
            "success": self.episode_success,
            "instruction": episode_data["instruction"],
            "goals": episode_data["goals"]
        }
    
    def get_metrics(self) -> Dict[str, float]:
        """Get episode metrics"""
        return {
            "success": 1.0 if self.episode_success else 0.0,
            "spl": 0.8 if self.episode_success else 0.0,
            "ndtw": 0.7 if self.episode_success else 0.3,
            "sdtw": 0.6 if self.episode_success else 0.2,
            "path_length": float(self.episode_step_count),
            "collisions": {"count": 1 if self._check_collision() else 0}
        }
    
    def close(self):
        """Close environment"""
        self.current_episode_id = None
        self.episode_step_count = 0
        self.episode_reward = 0.0
        self.episode_success = False
    
    def render(self, mode: str = "rgb") -> np.ndarray:
        """Render current observation"""
        if mode == "rgb":
            return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        elif mode == "depth":
            return np.random.randint(0, 255, (224, 224, 1), dtype=np.uint8)
        else:
            return np.zeros((224, 224, 3), dtype=np.uint8)
