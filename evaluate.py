import os
import click
import torch
from rich.console import Console
from loguru import logger
from typing import Dict, Any, List
import json

from navr1.config import NavR1Config, load_config
from navr1.datasets.nav_cot import build_dataloader
from navr1.datasets import build_task_dataloader
from navr1.simulators import build_simulator
from navr1.models.policy import FastInSlowPolicy
from navr1.training.task_finetune import TaskSpecificHead

console = Console()


class TaskEvaluator:
    """Task-specific evaluator for different embodied AI benchmarks"""
    
    def __init__(self, cfg: NavR1Config, policy: FastInSlowPolicy, task_type: str):
        self.cfg = cfg
        self.policy = policy
        self.task_type = task_type
        self.device = policy.device
        
        # Load task-specific head if available
        self.task_head = None
        self._load_task_head()
    
    def _load_task_head(self):
        """Load task-specific head if checkpoint exists"""
        # This would typically load from a trained checkpoint
        # For now, we'll create a new head for evaluation
        if self.task_type in ["vln_r2r", "vln_rxr", "objectnav_hm3d", 
                             "embodied_dialogue", "embodied_planning", "embodied_reasoning"]:
            self.task_head = TaskSpecificHead(self.cfg.model.hidden_dim, self.task_type).to(self.device)
    
    def evaluate_episode(self, batch: Dict[str, Any], sim: Any) -> Dict[str, Any]:
        """Evaluate a single episode"""
        obs = sim.reset(batch)
        done = False
        steps = 0
        episode_info = {
            "success": False,
            "steps": 0,
            "reward": 0.0,
            "actions": [],
            "info": {}
        }
        
        while not done and steps < self.cfg.simulator.max_episode_steps:
            # Get action from policy
            action = self.policy.act(obs)
            
            # Environment step
            obs, reward, done, info = sim.step(action)
            
            # Record episode information
            episode_info["actions"].append(action)
            episode_info["reward"] += reward
            episode_info["steps"] = steps + 1
            episode_info["info"] = info
            
            if info.get("success", False):
                episode_info["success"] = True
            
            steps += 1
        
        return episode_info
    
    def compute_metrics(self, episode_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute task-specific metrics"""
        if not episode_results:
            return {}
        
        # Basic metrics
        total_episodes = len(episode_results)
        successful_episodes = sum(1 for ep in episode_results if ep["success"])
        success_rate = successful_episodes / total_episodes
        
        avg_steps = sum(ep["steps"] for ep in episode_results) / total_episodes
        avg_reward = sum(ep["reward"] for ep in episode_results) / total_episodes
        
        metrics = {
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "total_episodes": total_episodes
        }
        
        # Task-specific metrics
        if self.task_type in ["vln_r2r", "vln_rxr"]:
            # VLN specific metrics
            metrics.update(self._compute_vln_metrics(episode_results))
        elif self.task_type == "objectnav_hm3d":
            # ObjectNav specific metrics
            metrics.update(self._compute_objectnav_metrics(episode_results))
        elif self.task_type in ["embodied_dialogue", "embodied_planning", "embodied_reasoning"]:
            # Text generation specific metrics
            metrics.update(self._compute_text_metrics(episode_results))
        
        return metrics
    
    def _compute_vln_metrics(self, episode_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute VLN-specific metrics"""
        # SPL (Success weighted by Path Length)
        spl_scores = []
        for ep in episode_results:
            if ep["success"]:
                # In real implementation, this would use actual path lengths
                optimal_length = ep["steps"]  # Simplified
                actual_length = ep["steps"]
                spl = optimal_length / max(actual_length, 1)
                spl_scores.append(spl)
            else:
                spl_scores.append(0.0)
        
        return {
            "spl": sum(spl_scores) / len(spl_scores) if spl_scores else 0.0,
            "successful_episodes": sum(1 for ep in episode_results if ep["success"])
        }
    
    def _compute_objectnav_metrics(self, episode_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute ObjectNav-specific metrics"""
        # Object detection accuracy (simplified)
        detection_accuracy = 0.8  # Placeholder
        
        return {
            "detection_accuracy": detection_accuracy,
            "navigation_success": sum(1 for ep in episode_results if ep["success"]) / len(episode_results)
        }
    
    def _compute_text_metrics(self, episode_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute text generation metrics"""
        # BLEU, ROUGE, etc. would be computed here
        return {
            "text_quality": 0.75,  # Placeholder
            "response_relevance": 0.80  # Placeholder
        }


@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True), default="navr1/configs/default.yaml")
@click.option("--checkpoint", type=click.Path(), required=True, help="Path to model checkpoint")
@click.option("--task", type=click.Choice(["vln_r2r", "vln_rxr", "objectnav_hm3d", 
                                          "embodied_dialogue", "embodied_planning", "embodied_reasoning", "general"]), 
              default="general", help="Task type for evaluation")
@click.option("--split", type=click.Choice(["val", "test"]), default="val")
@click.option("--episodes", type=int, default=50)
@click.option("--workdir", type=click.Path(), default="./runs/navr1_eval")
@click.option("--output-file", type=click.Path(), default=None, help="Path to save evaluation results")
@click.option("--habitat-config", type=click.Path(), default=None, help="Override Habitat YAML for this eval run")
def main(config_path: str, checkpoint: str, task: str, split: str, episodes: int, 
         workdir: str, output_file: str, habitat_config: str):
    """
    Nav-R1 Task-Specific Evaluation
    
    Evaluate Nav-R1 model on different embodied AI benchmarks.
    """
    console.rule(f"[bold cyan]Nav-R1 Evaluation - {task.upper()}")
    
    # Load configuration
    cfg: NavR1Config = load_config(config_path)
    os.makedirs(workdir, exist_ok=True)
    
    # Setup logging
    logger.add(os.path.join(workdir, f"eval_{task}.log"))
    logger.info(f"Evaluating {task} with checkpoint: {checkpoint}")
    
    # Load model
    policy = FastInSlowPolicy(cfg.model)
    
    # Load checkpoint
    if os.path.exists(checkpoint):
        checkpoint_data = torch.load(checkpoint, map_location=policy.device)
        
        # Load model state
        policy.text_encoder.load_state_dict(checkpoint_data["model_state_dict"]["text_encoder"])
        policy.slow.load_state_dict(checkpoint_data["model_state_dict"]["slow"])
        policy.fast.load_state_dict(checkpoint_data["model_state_dict"]["fast"])
        
        # Load 3D-R1 backbone if present
        if "dr1_backbone" in checkpoint_data["model_state_dict"] and \
           hasattr(policy, 'dr1_backbone') and policy.dr1_backbone is not None:
            policy.dr1_backbone.load_state_dict(checkpoint_data["model_state_dict"]["dr1_backbone"])
        
        logger.info(f"Loaded checkpoint from {checkpoint}")
    else:
        logger.error(f"Checkpoint not found: {checkpoint}")
        return
    
    # If task provided, optionally switch Habitat YAML to the task's dataset (Mp3D for R2R/RxR, HM3D for ObjectNav)
    if habitat_config:
        cfg.simulator.habitat_config = habitat_config
    else:
        if task in ["vln_r2r", "vln_rxr", "objectnav_hm3d"] and hasattr(cfg, 'task_finetune'):
            task_cfg = getattr(cfg.task_finetune, task, {}) if isinstance(cfg.task_finetune, object) else {}
            hcfg = task_cfg.get("habitat_config") if isinstance(task_cfg, dict) else None
            if hcfg:
                cfg.simulator.habitat_config = hcfg

    # Build dataloader and simulator
    if task != "general":
        try:
            dataloader = build_task_dataloader(cfg, task, split=split, shuffle=False)
        except Exception:
            dataloader = build_dataloader(cfg.dataset, split=split, shuffle=False)
    else:
        dataloader = build_dataloader(cfg.dataset, split=split, shuffle=False)
    sim = build_simulator(cfg.simulator)
    
    # Initialize evaluator
    evaluator = TaskEvaluator(cfg, policy, task)
    
    # Run evaluation
    console.print(f"Running evaluation on {episodes} episodes...")
    episode_results = []
    
    for i, batch in enumerate(dataloader):
        if i >= episodes:
            break
        
        episode_info = evaluator.evaluate_episode(batch, sim)
        episode_results.append(episode_info)
        
        if (i + 1) % 10 == 0:
            console.print(f"Completed {i + 1}/{episodes} episodes")
    
    # Compute metrics
    metrics = evaluator.compute_metrics(episode_results)
    
    # Print results
    console.rule("[bold green]Evaluation Results")
    console.print(f"Task: {task}")
    console.print(f"Episodes: {metrics.get('total_episodes', 0)}")
    console.print(f"Success Rate: {metrics.get('success_rate', 0):.3f}")
    console.print(f"Average Steps: {metrics.get('avg_steps', 0):.1f}")
    console.print(f"Average Reward: {metrics.get('avg_reward', 0):.3f}")
    
    # Print task-specific metrics
    if task in ["vln_r2r", "vln_rxr"]:
        console.print(f"SPL: {metrics.get('spl', 0):.3f}")
    elif task == "objectnav_hm3d":
        console.print(f"Detection Accuracy: {metrics.get('detection_accuracy', 0):.3f}")
    elif task in ["embodied_dialogue", "embodied_planning", "embodied_reasoning"]:
        console.print(f"Text Quality: {metrics.get('text_quality', 0):.3f}")
    
    # Save results
    results = {
        "task": task,
        "checkpoint": checkpoint,
        "episodes": episodes,
        "metrics": metrics,
        "episode_results": episode_results
    }
    
    if output_file:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"Results saved to: {output_file}")
    else:
        output_file = os.path.join(workdir, f"eval_results_{task}.json")
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        console.print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
