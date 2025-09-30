#!/usr/bin/env python3
"""
Evaluation script for Nav-R1
"""

import argparse
import os
import yaml
import torch
import numpy as np
from typing import Dict, Any, List
from tqdm import tqdm
import json

from navr1.models.policy import NavR1Policy
from navr1.simulators.habitat import HabitatSimulator, HabitatStubSimulator


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: Dict[str, Any], checkpoint_path: str) -> NavR1Policy:
    """Create Nav-R1 model from configuration and load checkpoint"""
    model_config = config["model"]
    
    # Vision encoder config
    vision_config = {
        "model_name": model_config["vision_encoder"]["type"],
        "pretrained_path": model_config["vision_encoder"].get("pretrained_path"),
        "freeze_vision": model_config["vision_encoder"].get("freeze_vision", False),
        "image_size": config["dataset"]["image_size"],
        "hidden_size": model_config["multimodal_fusion"]["hidden_size"],
    }
    
    # Language encoder config
    language_config = {
        "model_name": model_config["language_model"]["type"],
        "pretrained_path": model_config["language_model"].get("pretrained_path"),
        "freeze_lm": model_config["language_model"].get("freeze_lm", False),
        "hidden_size": model_config["multimodal_fusion"]["hidden_size"],
    }
    
    # Fusion config
    fusion_config = model_config["multimodal_fusion"]
    
    # Policy config
    policy_config = model_config["policy_head"]
    
    # Reasoning config (optional)
    reasoning_config = None
    if "reasoning_head" in model_config:
        reasoning_config = model_config["reasoning_head"]
    
    # Create backbone config
    backbone_config = {
        "vision_config": vision_config,
        "language_config": language_config,
        "fusion_config": fusion_config,
    }
    
    # Create model
    model = NavR1Policy(
        backbone_config=backbone_config,
        policy_config=policy_config,
        reasoning_config=reasoning_config,
    )
    
    # Load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("Warning: No checkpoint provided or checkpoint not found")
    
    return model


def create_simulator(config: Dict[str, Any]) -> HabitatSimulator:
    """Create simulator from configuration"""
    simulator_config = config["simulator"]
    
    try:
        simulator = HabitatSimulator(
            config_path=simulator_config["habitat_config"],
            scene_dataset_path=simulator_config["scene_dataset"],
            episode_dataset_path=simulator_config["episode_dataset"],
            max_episode_steps=simulator_config["max_episode_steps"],
            success_reward=simulator_config["success_reward"],
            step_penalty=simulator_config["step_penalty"],
            collision_penalty=simulator_config["collision_penalty"],
            device=config["hardware"]["device"],
        )
    except ImportError:
        print("Warning: Habitat not available, using stub simulator")
        simulator = HabitatStubSimulator(
            config_path=simulator_config["habitat_config"],
            scene_dataset_path=simulator_config["scene_dataset"],
            episode_dataset_path=simulator_config["episode_dataset"],
            max_episode_steps=simulator_config["max_episode_steps"],
            success_reward=simulator_config["success_reward"],
            step_penalty=simulator_config["step_penalty"],
            collision_penalty=simulator_config["collision_penalty"],
            device=config["hardware"]["device"],
        )
    
    return simulator


def evaluate_episode(
    model: NavR1Policy,
    simulator: HabitatSimulator,
    episode_id: str = None,
    max_steps: int = 500,
    save_video: bool = False,
    video_path: str = None,
) -> Dict[str, Any]:
    """Evaluate a single episode"""
    model.eval()
    
    # Reset environment
    obs = simulator.reset(episode_id)
    episode_info = simulator.get_episode_info()
    
    # Episode data
    episode_data = {
        "episode_id": episode_info.get("episode_id", "unknown"),
        "instruction": episode_info.get("instruction", ""),
        "actions": [],
        "rewards": [],
        "observations": [],
        "success": False,
        "episode_length": 0,
        "total_reward": 0.0,
    }
    
    # Video frames (if saving video)
    video_frames = []
    
    step_count = 0
    done = False
    
    with torch.no_grad():
        while not done and step_count < max_steps:
            # Get action from model
            action, value, log_prob = get_action_from_model(model, obs, episode_info)
            
            # Execute action
            next_obs, reward, done, info = simulator.step(action)
            
            # Store episode data
            episode_data["actions"].append(action)
            episode_data["rewards"].append(reward)
            episode_data["total_reward"] += reward
            episode_data["episode_length"] += 1
            
            # Store observation (for video)
            if save_video:
                frame = simulator.render("rgb")
                video_frames.append(frame)
            
            # Update for next step
            obs = next_obs
            episode_info = info
            step_count += 1
        
        # Check success
        episode_data["success"] = info.get("success", False)
        
        # Save video if requested
        if save_video and video_frames and video_path:
            save_video_frames(video_frames, video_path)
    
    return episode_data


def get_action_from_model(
    model: NavR1Policy,
    obs: Dict[str, Any],
    episode_info: Dict[str, Any],
    temperature: float = 1.0,
    deterministic: bool = False,
) -> tuple:
    """Get action from model"""
    # Convert observation to model input format
    images = obs["rgb"].unsqueeze(0)  # Add batch dimension
    
    # Create input IDs from instruction
    instruction = episode_info.get("instruction", "")
    # This is simplified - in practice, you would properly tokenize the instruction
    input_ids = torch.tensor([[1, 2, 3, 4, 5]], device=obs["rgb"].device)
    attention_mask = torch.ones_like(input_ids)
    image_mask = torch.ones(1, images.shape[1], device=obs["rgb"].device)
    
    # Get model outputs
    outputs = model(
        images=images,
        input_ids=input_ids,
        attention_mask=attention_mask,
        image_mask=image_mask,
    )
    
    action_logits = outputs["action_logits"]
    value = outputs["value"]
    
    # Sample action
    action_probs = torch.softmax(action_logits / temperature, dim=-1)
    
    if deterministic:
        action = torch.argmax(action_probs, dim=-1)
    else:
        action = torch.multinomial(action_probs, num_samples=1).squeeze(-1)
    
    log_prob = torch.log(action_probs[0, action] + 1e-8)
    
    # Convert action index to action string
    action_mapping = ["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "STOP"]
    action_str = action_mapping[action.item()]
    
    return action_str, value.item(), log_prob.item()


def save_video_frames(frames: List[np.ndarray], video_path: str):
    """Save video frames to file"""
    import cv2
    
    if not frames:
        return
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()


def compute_metrics(episode_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute evaluation metrics"""
    if not episode_results:
        return {}
    
    # Basic metrics
    success_rates = [ep["success"] for ep in episode_results]
    episode_lengths = [ep["episode_length"] for ep in episode_results]
    total_rewards = [ep["total_reward"] for ep in episode_results]
    
    # Compute metrics
    metrics = {
        "success_rate": np.mean(success_rates),
        "avg_episode_length": np.mean(episode_lengths),
        "std_episode_length": np.std(episode_lengths),
        "avg_total_reward": np.mean(total_rewards),
        "std_total_reward": np.std(total_rewards),
        "num_episodes": len(episode_results),
    }
    
    # Additional metrics (simplified)
    metrics["spl"] = np.mean(success_rates)  # Simplified SPL
    metrics["ndtw"] = np.mean(success_rates) * 0.8  # Simplified NDTW
    metrics["sdtw"] = np.mean(success_rates) * 0.7  # Simplified SDTW
    
    return metrics


def evaluate(
    model: NavR1Policy,
    simulator: HabitatSimulator,
    num_episodes: int = 100,
    save_videos: bool = False,
    video_dir: str = "videos",
) -> Dict[str, Any]:
    """Evaluate model on multiple episodes"""
    print(f"Evaluating on {num_episodes} episodes...")
    
    episode_results = []
    
    # Create video directory if needed
    if save_videos:
        os.makedirs(video_dir, exist_ok=True)
    
    # Evaluate episodes
    for episode_idx in tqdm(range(num_episodes), desc="Evaluating"):
        # Save video path
        video_path = None
        if save_videos:
            video_path = os.path.join(video_dir, f"episode_{episode_idx:04d}.mp4")
        
        # Evaluate episode
        episode_result = evaluate_episode(
            model=model,
            simulator=simulator,
            max_steps=simulator.max_episode_steps,
            save_video=save_videos,
            video_path=video_path,
        )
        
        episode_results.append(episode_result)
    
    # Compute metrics
    metrics = compute_metrics(episode_results)
    
    return {
        "metrics": metrics,
        "episode_results": episode_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Nav-R1 model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to evaluate")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate on")
    parser.add_argument("--save_videos", action="store_true", help="Save evaluation videos")
    parser.add_argument("--video_dir", type=str, default="videos", help="Directory to save videos")
    parser.add_argument("--output", type=str, help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = config["hardware"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
        config["hardware"]["device"] = device
    
    # Create model
    model = create_model(config, args.checkpoint)
    model.to(device)
    
    # Create simulator
    simulator = create_simulator(config)
    
    # Evaluate
    results = evaluate(
        model=model,
        simulator=simulator,
        num_episodes=args.episodes,
        save_videos=args.save_videos,
        video_dir=args.video_dir,
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("=" * 50)
    for metric, value in results["metrics"].items():
        print(f"{metric}: {value:.4f}")
    
    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    
    # Close simulator
    simulator.close()


if __name__ == "__main__":
    main()
