"""
Metrics computation utilities for Nav-R1
"""

import numpy as np
from typing import Dict, List, Any, Tuple
import torch


def compute_navigation_metrics(
    episode_results: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Compute navigation-specific metrics"""
    
    if not episode_results:
        return {}
    
    # Extract metrics
    success_rates = [ep.get("success", False) for ep in episode_results]
    episode_lengths = [ep.get("episode_length", 0) for ep in episode_results]
    total_rewards = [ep.get("total_reward", 0.0) for ep in episode_results]
    
    # Basic metrics
    success_rate = np.mean(success_rates)
    avg_episode_length = np.mean(episode_lengths)
    std_episode_length = np.std(episode_lengths)
    avg_total_reward = np.mean(total_rewards)
    std_total_reward = np.std(total_rewards)
    
    # Navigation-specific metrics (simplified implementations)
    spl = compute_spl(episode_results)
    ndtw = compute_ndtw(episode_results)
    sdtw = compute_sdtw(episode_results)
    
    metrics = {
        "success_rate": success_rate,
        "avg_episode_length": avg_episode_length,
        "std_episode_length": std_episode_length,
        "avg_total_reward": avg_total_reward,
        "std_total_reward": std_total_reward,
        "spl": spl,
        "ndtw": ndtw,
        "sdtw": sdtw,
        "num_episodes": len(episode_results),
    }
    
    return metrics


def compute_spl(episode_results: List[Dict[str, Any]]) -> float:
    """Compute Success weighted by Path Length (SPL)"""
    if not episode_results:
        return 0.0
    
    spl_scores = []
    for ep in episode_results:
        success = ep.get("success", False)
        episode_length = ep.get("episode_length", 1)
        
        # Simplified SPL calculation
        # In practice, you would use the actual shortest path length
        shortest_path_length = max(1, episode_length // 2)  # Simplified
        
        if success:
            spl_score = shortest_path_length / max(episode_length, 1)
        else:
            spl_score = 0.0
            
        spl_scores.append(spl_score)
    
    return np.mean(spl_scores)


def compute_ndtw(episode_results: List[Dict[str, Any]]) -> float:
    """Compute Normalized Dynamic Time Warping (NDTW)"""
    if not episode_results:
        return 0.0
    
    ndtw_scores = []
    for ep in episode_results:
        success = ep.get("success", False)
        episode_length = ep.get("episode_length", 1)
        
        # Simplified NDTW calculation
        # In practice, you would compute the actual DTW distance
        if success:
            ndtw_score = 1.0 - (episode_length - 1) / (episode_length + 1)
        else:
            ndtw_score = 0.0
            
        ndtw_scores.append(ndtw_score)
    
    return np.mean(ndtw_scores)


def compute_sdtw(episode_results: List[Dict[str, Any]]) -> float:
    """Compute Success weighted by Dynamic Time Warping (SDTW)"""
    if not episode_results:
        return 0.0
    
    sdtw_scores = []
    for ep in episode_results:
        success = ep.get("success", False)
        episode_length = ep.get("episode_length", 1)
        
        # Simplified SDTW calculation
        # In practice, you would compute the actual DTW distance
        if success:
            sdtw_score = 1.0 - (episode_length - 1) / (episode_length + 1)
        else:
            sdtw_score = 0.0
            
        sdtw_scores.append(sdtw_score)
    
    return np.mean(sdtw_scores)


def compute_reasoning_metrics(
    reasoning_outputs: List[Dict[str, Any]]
) -> Dict[str, float]:
    """Compute reasoning-specific metrics"""
    
    if not reasoning_outputs:
        return {}
    
    # Extract reasoning components
    observations = [ro.get("observation", "") for ro in reasoning_outputs]
    reasoning_steps = [ro.get("reasoning_steps", []) for ro in reasoning_outputs]
    conclusions = [ro.get("conclusion", "") for ro in reasoning_outputs]
    confidences = [ro.get("confidence", 0.0) for ro in reasoning_outputs]
    
    # Compute metrics
    avg_confidence = np.mean(confidences)
    std_confidence = np.std(confidences)
    
    # Reasoning structure metrics
    has_observation = np.mean([len(obs) > 0 for obs in observations])
    has_reasoning_steps = np.mean([len(steps) > 0 for steps in reasoning_steps])
    has_conclusion = np.mean([len(conc) > 0 for conc in conclusions])
    
    # Reasoning quality metrics (simplified)
    avg_reasoning_steps = np.mean([len(steps) for steps in reasoning_steps])
    avg_observation_length = np.mean([len(obs) for obs in observations])
    avg_conclusion_length = np.mean([len(conc) for conc in conclusions])
    
    metrics = {
        "avg_confidence": avg_confidence,
        "std_confidence": std_confidence,
        "has_observation": has_observation,
        "has_reasoning_steps": has_reasoning_steps,
        "has_conclusion": has_conclusion,
        "avg_reasoning_steps": avg_reasoning_steps,
        "avg_observation_length": avg_observation_length,
        "avg_conclusion_length": avg_conclusion_length,
        "num_reasoning_outputs": len(reasoning_outputs),
    }
    
    return metrics


def compute_composite_metrics(
    navigation_metrics: Dict[str, float],
    reasoning_metrics: Dict[str, float],
    weights: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """Compute composite metrics combining navigation and reasoning"""
    
    if weights is None:
        weights = {
            "navigation": 0.7,
            "reasoning": 0.3,
        }
    
    # Normalize metrics to [0, 1] range
    nav_score = (
        navigation_metrics.get("success_rate", 0.0) * 0.4 +
        navigation_metrics.get("spl", 0.0) * 0.3 +
        navigation_metrics.get("ndtw", 0.0) * 0.3
    )
    
    reasoning_score = (
        reasoning_metrics.get("avg_confidence", 0.0) * 0.4 +
        reasoning_metrics.get("has_reasoning_steps", 0.0) * 0.3 +
        reasoning_metrics.get("avg_reasoning_steps", 0.0) / 10.0 * 0.3  # Normalize to [0, 1]
    )
    
    # Composite score
    composite_score = (
        weights["navigation"] * nav_score +
        weights["reasoning"] * reasoning_score
    )
    
    metrics = {
        "composite_score": composite_score,
        "navigation_score": nav_score,
        "reasoning_score": reasoning_score,
        "navigation_metrics": navigation_metrics,
        "reasoning_metrics": reasoning_metrics,
    }
    
    return metrics


def compute_training_metrics(
    losses: Dict[str, float],
    learning_rate: float,
    epoch: int,
    step: int
) -> Dict[str, float]:
    """Compute training-specific metrics"""
    
    metrics = {
        "epoch": epoch,
        "step": step,
        "learning_rate": learning_rate,
    }
    
    # Add loss metrics
    for key, value in losses.items():
        metrics[f"loss_{key}"] = value
    
    # Compute loss ratios
    if "total_loss" in losses:
        total_loss = losses["total_loss"]
        for key, value in losses.items():
            if key != "total_loss" and value > 0:
                metrics[f"loss_ratio_{key}"] = value / total_loss
    
    return metrics


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> str:
    """Format metrics for display"""
    
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{key}: {value:.{precision}f}")
        else:
            formatted.append(f"{key}: {value}")
    
    return ", ".join(formatted)
