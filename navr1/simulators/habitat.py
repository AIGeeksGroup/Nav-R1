from typing import Any, Dict, Tuple
from dataclasses import dataclass

from loguru import logger
from navr1.config import SimulatorConfig

try:
    import habitat
    from habitat.config.default import get_config
except Exception:  # pragma: no cover
    habitat = None
    get_config = None


def extract_success(info: Dict[str, Any]) -> bool:
    # Common places Habitat tasks record success
    # VLN / R2R often uses success under metrics
    metrics = info.get("metrics") if isinstance(info, dict) else None
    if isinstance(metrics, dict) and "success" in metrics:
        return bool(metrics.get("success", 0) >= 1)
    # ObjectNav HM3D
    if "success" in info:
        return bool(info.get("success"))
    # Fallback
    return False


@dataclass
class HabitatSimulator:
    cfg: SimulatorConfig

    def __post_init__(self):
        if habitat is None or get_config is None:
            raise ImportError("Habitat-Lab not installed. Please install habitat-lab and habitat-sim.")
        if not self.cfg.habitat_config:
            raise ValueError("simulator.habitat_config must be provided for HabitatSimulator")
        self._hcfg = get_config(self.cfg.habitat_config)
        self.env = habitat.Env(config=self._hcfg)
        self._steps = 0

    def reset(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self._steps = 0
        obs = self.env.reset()
        return {"obs": obs, "batch": batch}

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self._steps += 1
        act_str = str(action)
        act_name = self.cfg.action_map.get(act_str, act_str)
        if act_name not in self.env.action_space:
            logger.warning(f"Unknown action '{act_name}' for Habitat action space. Defaulting to 'stop'.")
            act_name = "stop" if "stop" in self.env.action_space else list(self.env.action_space.keys())[0]
        out = self.env.step(act_name)
        obs = out.observations if hasattr(out, "observations") else out[0] if isinstance(out, tuple) else out
        reward = out.reward if hasattr(out, "reward") else 0.0
        done = out.done if hasattr(out, "done") else (self._steps >= self.cfg.max_episode_steps)
        info = out.info if hasattr(out, "info") else {}
        if self._steps >= self.cfg.max_episode_steps:
            done = True
        info = {**info, "success": extract_success(info)}
        return {"obs": obs}, float(reward), bool(done), info
