from typing import Any, Dict, Tuple
from dataclasses import dataclass

from loguru import logger
from navr1.config import SimulatorConfig


@dataclass
class HabitatStub:
    cfg: SimulatorConfig

    def __post_init__(self):
        logger.warning("Habitat-Lab/Sim not installed. Using stub.")

    def reset(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {"obs": "habitat_reset", "batch": batch}

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        return {"obs": "habitat_step"}, 0.0, True, {"success": False}
