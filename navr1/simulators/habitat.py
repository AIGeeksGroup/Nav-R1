"""
Habitat Simulator Interface for Nav-R1
"""

import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
import cv2
from PIL import Image
import json
import os

try:
    import habitat
    from habitat import make_dataset
    from habitat.config import read_write
    from habitat.core.env import Env
    from habitat.core.registry import registry
    from habitat.core.simulator import Observations
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
    from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal
    from habitat.tasks.vln.vln import VLNEpisode
    HABITAT_AVAILABLE = True
except ImportError:
    HABITAT_AVAILABLE = False
    print("Warning: Habitat-Lab not available. Using stub simulator.")


class HabitatSimulator:
    """
    Habitat simulator wrapper for Nav-R1
    """
    
    def __init__(
        self,
        config_path: str,
        scene_dataset_path: str,
        episode_dataset_path: str,
        max_episode_steps: int = 500,
        success_reward: float = 10.0,
        step_penalty: float = -0.01,
        collision_penalty: float = -1.0,
        device: str = "cuda"
    ):
        if not HABITAT_AVAILABLE:
            raise ImportError("Habitat-Lab is required but not installed")
            
        self.config_path = config_path
        self.scene_dataset_path = scene_dataset_path
        self.episode_dataset_path = episode_dataset_path
        self.max_episode_steps = max_episode_steps
        self.success_reward = success_reward
        self.step_penalty = step_penalty
        self.collision_penalty = collision_penalty
        self.device = device
        
        # Initialize environment
        self.env = None
        self.current_episode = None
        self.episode_step_count = 0
        self.episode_reward = 0.0
        self.episode_success = False
        
        # Action mapping
        self.action_mapping = {
            "MOVE_FORWARD": HabitatSimActions.MOVE_FORWARD,
            "TURN_LEFT": HabitatSimActions.TURN_LEFT,
            "TURN_RIGHT": HabitatSimActions.TURN_RIGHT,
            "STOP": HabitatSimActions.STOP,
        }
        
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup Habitat environment"""
        # Load configuration
        config = habitat.get_config(self.config_path)
        
        # Update paths
        with read_write(config):
            config.habitat.dataset.data_path = self.episode_dataset_path
            config.habitat.dataset.scenes_dir = self.scene_dataset_path
            config.habitat.environment.max_episode_steps = self.max_episode_steps
            
        # Create environment
        self.env = Env(config=config)
        
    def reset(self, episode_id: Optional[str] = None) -> Dict[str, Any]:
        """Reset environment to new episode"""
        if episode_id is not None:
            # Find specific episode
            episode = self.env._dataset.get_episode(episode_id)
            self.env._current_episode = episode
        else:
            # Random episode
            self.env.reset()
            
        self.current_episode = self.env.current_episode
        self.episode_step_count = 0
        self.episode_reward = 0.0
        self.episode_success = False
        
        # Get initial observations
        observations = self.env.step(HabitatSimActions.STOP)
        
        return self._process_observations(observations)
    
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute action and return next state"""
        if action not in self.action_mapping:
            raise ValueError(f"Unknown action: {action}")
            
        habitat_action = self.action_mapping[action]
        observations = self.env.step(habitat_action)
        
        self.episode_step_count += 1
        
        # Calculate reward
        reward = self._calculate_reward(observations, action)
        self.episode_reward += reward
        
        # Check if episode is done
        done = self._is_episode_done(observations)
        
        # Process observations
        processed_obs = self._process_observations(observations)
        
        # Create info dict
        info = {
            "episode_id": self.current_episode.episode_id,
            "step_count": self.episode_step_count,
            "episode_reward": self.episode_reward,
            "success": self.episode_success,
            "collision": self._check_collision(observations),
            "action": action
        }
        
        return processed_obs, reward, done, info
    
    def _process_observations(self, observations: Observations) -> Dict[str, Any]:
        """Process Habitat observations into Nav-R1 format"""
        processed = {}
        
        # RGB image
        if "rgb" in observations:
            rgb = observations["rgb"]
            if isinstance(rgb, np.ndarray):
                rgb = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            processed["rgb"] = rgb.to(self.device)
            
        # Depth image
        if "depth" in observations:
            depth = observations["depth"]
            if isinstance(depth, np.ndarray):
                depth = torch.from_numpy(depth).unsqueeze(0).float()
            processed["depth"] = depth.to(self.device)
            
        # Agent state
        if "agent_state" in observations:
            agent_state = observations["agent_state"]
            processed["agent_state"] = {
                "position": torch.tensor(agent_state.position, device=self.device),
                "rotation": torch.tensor(agent_state.rotation, device=self.device),
                "velocity": torch.tensor(agent_state.velocity, device=self.device) if hasattr(agent_state, 'velocity') else None
            }
            
        # GPS and compass
        if "gps" in observations:
            processed["gps"] = torch.tensor(observations["gps"], device=self.device)
        if "compass" in observations:
            processed["compass"] = torch.tensor(observations["compass"], device=self.device)
            
        # Add episode information
        if self.current_episode is not None:
            processed["episode_info"] = {
                "episode_id": self.current_episode.episode_id,
                "instruction": getattr(self.current_episode, 'instruction', {}).get('instruction_text', ''),
                "goals": [goal.position for goal in getattr(self.current_episode, 'goals', [])],
                "scene_id": self.current_episode.scene_id
            }
            
        return processed
    
    def _calculate_reward(self, observations: Observations, action: str) -> float:
        """Calculate reward for current step"""
        reward = self.step_penalty
        
        # Success reward
        if self._check_success(observations):
            reward += self.success_reward
            self.episode_success = True
            
        # Collision penalty
        if self._check_collision(observations):
            reward += self.collision_penalty
            
        return reward
    
    def _check_success(self, observations: Observations) -> bool:
        """Check if episode is successful"""
        if hasattr(self.env, 'get_metrics'):
            metrics = self.env.get_metrics()
            return metrics.get("success", 0.0) > 0.0
        return False
    
    def _check_collision(self, observations: Observations) -> bool:
        """Check if agent collided"""
        if hasattr(self.env, 'get_metrics'):
            metrics = self.env.get_metrics()
            return metrics.get("collisions", {}).get("count", 0) > 0
        return False
    
    def _is_episode_done(self, observations: Observations) -> bool:
        """Check if episode is done"""
        # Check if max steps reached
        if self.episode_step_count >= self.max_episode_steps:
            return True
            
        # Check if episode is successful
        if self.episode_success:
            return True
            
        # Check if environment says episode is done
        if hasattr(self.env, 'episode_over'):
            return self.env.episode_over
            
        return False
    
    def get_episode_info(self) -> Dict[str, Any]:
        """Get current episode information"""
        if self.current_episode is None:
            return {}
            
        info = {
            "episode_id": self.current_episode.episode_id,
            "scene_id": self.current_episode.scene_id,
            "step_count": self.episode_step_count,
            "episode_reward": self.episode_reward,
            "success": self.episode_success
        }
        
        # Add task-specific information
        if hasattr(self.current_episode, 'instruction'):
            info["instruction"] = self.current_episode.instruction.get('instruction_text', '')
            
        if hasattr(self.current_episode, 'goals'):
            info["goals"] = [goal.position for goal in self.current_episode.goals]
            
        if hasattr(self.current_episode, 'object_category'):
            info["object_category"] = self.current_episode.object_category
            
        return info
    
    def get_metrics(self) -> Dict[str, float]:
        """Get episode metrics"""
        if hasattr(self.env, 'get_metrics'):
            return self.env.get_metrics()
        return {}
    
    def close(self):
        """Close environment"""
        if self.env is not None:
            self.env.close()
            self.env = None
    
    def render(self, mode: str = "rgb") -> np.ndarray:
        """Render current observation"""
        if self.env is None:
            return np.zeros((224, 224, 3), dtype=np.uint8)
            
        observations = self.env.sim.get_sensor_observations()
        
        if mode == "rgb" and "rgb" in observations:
            return observations["rgb"]
        elif mode == "depth" and "depth" in observations:
            return observations["depth"]
        else:
            return np.zeros((224, 224, 3), dtype=np.uint8)


class HabitatStubSimulator:
    """
    Stub simulator for when Habitat is not available
    """
    
    def __init__(self, *args, **kwargs):
        self.device = kwargs.get("device", "cuda")
        self.max_episode_steps = kwargs.get("max_episode_steps", 500)
        self.episode_step_count = 0
        self.episode_reward = 0.0
        self.episode_success = False
        
    def reset(self, episode_id: Optional[str] = None) -> Dict[str, Any]:
        """Reset to dummy episode"""
        self.episode_step_count = 0
        self.episode_reward = 0.0
        self.episode_success = False
        
        return {
            "rgb": torch.zeros(3, 224, 224, device=self.device),
            "depth": torch.zeros(1, 224, 224, device=self.device),
            "agent_state": {
                "position": torch.zeros(3, device=self.device),
                "rotation": torch.zeros(4, device=self.device)
            },
            "episode_info": {
                "episode_id": episode_id or "dummy_episode",
                "instruction": "Dummy instruction",
                "goals": [],
                "scene_id": "dummy_scene"
            }
        }
    
    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Dummy step"""
        self.episode_step_count += 1
        
        reward = -0.01  # Step penalty
        done = self.episode_step_count >= self.max_episode_steps
        
        obs = self.reset()
        info = {
            "episode_id": "dummy_episode",
            "step_count": self.episode_step_count,
            "episode_reward": self.episode_reward,
            "success": False,
            "collision": False,
            "action": action
        }
        
        return obs, reward, done, info
    
    def get_episode_info(self) -> Dict[str, Any]:
        """Get dummy episode info"""
        return {
            "episode_id": "dummy_episode",
            "scene_id": "dummy_scene",
            "step_count": self.episode_step_count,
            "episode_reward": self.episode_reward,
            "success": self.episode_success
        }
    
    def get_metrics(self) -> Dict[str, float]:
        """Get dummy metrics"""
        return {"success": 0.0, "spl": 0.0}
    
    def close(self):
        """Dummy close"""
        pass
    
    def render(self, mode: str = "rgb") -> np.ndarray:
        """Dummy render"""
        return np.zeros((224, 224, 3), dtype=np.uint8)
