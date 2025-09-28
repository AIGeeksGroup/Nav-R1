from typing import Any

from .habitat import HabitatSimulator
from navr1.config import SimulatorConfig


def build_simulator(cfg: SimulatorConfig) -> Any:
    return HabitatSimulator(cfg)
