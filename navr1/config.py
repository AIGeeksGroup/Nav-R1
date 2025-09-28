from typing import Optional, Dict, List
from pydantic import BaseModel
import yaml

class DatasetConfig(BaseModel):
    name: str = "Nav-CoT-110K"
    path: Optional[str] = None
    split: str = "train"
    batch_size: int = 4
    num_workers: int = 2
    shuffle: bool = True

class SimulatorConfig(BaseModel):
    name: str = "dummy"
    max_episode_steps: int = 64
    habitat_config: Optional[str] = None
    action_map: Dict[str, str] = {
        "forward": "move_forward",
        "left": "turn_left",
        "right": "turn_right",
        "back": "move_backward",
        "turn_left": "turn_left",
        "stop": "stop",
    }

class ModelConfig(BaseModel):
    vision_dim: int = 256
    text_dim: int = 256
    hidden_dim: int = 512
    action_dim: int = 6
    # 3D-R1 integration
    use_3dr1_backbone: bool = True
    dr1_model_path: Optional[str] = None
    freeze_3dr1_encoder: bool = False

class SFTConfig(BaseModel):
    learning_rate: float = 5e-5
    batch_size: int = 8
    epochs: int = 3
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    save_steps: int = 500

class RLConfig(BaseModel):
    gamma: float = 0.99
    lam: float = 0.95
    learning_rate: float = 3e-4
    batch_size: int = 8
    epochs: int = 3
    rollout_length: int = 64
    # Reward coefficients
    format_coef: float = 0.2
    understanding_coef: float = 0.3
    navigation_coef: float = 0.5

class TaskFinetuneConfig(BaseModel):
    # Task-specific configurations
    vln_r2r: Dict = {
        "learning_rate": 2e-5,
        "batch_size": 4,
        "epochs": 5,
        "dataset_path": None
    }
    vln_rxr: Dict = {
        "learning_rate": 2e-5,
        "batch_size": 4,
        "epochs": 5,
        "dataset_path": None
    }
    objectnav_hm3d: Dict = {
        "learning_rate": 2e-5,
        "batch_size": 4,
        "epochs": 5,
        "dataset_path": None
    }
    embodied_dialogue: Dict = {
        "learning_rate": 1e-5,
        "batch_size": 2,
        "epochs": 3,
        "dataset_path": None
    }
    embodied_planning: Dict = {
        "learning_rate": 1e-5,
        "batch_size": 2,
        "epochs": 3,
        "dataset_path": None
    }
    embodied_reasoning: Dict = {
        "learning_rate": 1e-5,
        "batch_size": 2,
        "epochs": 3,
        "dataset_path": None
    }

class TrainingStageConfig(BaseModel):
    stage: str = "sft"  # sft, rl, finetune
    checkpoint_path: Optional[str] = None
    output_dir: str = "./runs/navr1"
    save_best_model: bool = True
    early_stopping_patience: int = 3

class NavR1Config(BaseModel):
    dataset: DatasetConfig = DatasetConfig()
    simulator: SimulatorConfig = SimulatorConfig()
    model: ModelConfig = ModelConfig()
    sft: SFTConfig = SFTConfig()
    rl: RLConfig = RLConfig()
    task_finetune: TaskFinetuneConfig = TaskFinetuneConfig()
    training: TrainingStageConfig = TrainingStageConfig()


def load_config(path: str) -> NavR1Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return NavR1Config(**raw)
