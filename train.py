import os
import sys
import yaml
import click
from rich.console import Console
from loguru import logger

from navr1.config import NavR1Config, load_config
from navr1.datasets.nav_cot import build_dataloader
from navr1.simulators import build_simulator
from navr1.models.policy import FastInSlowPolicy
from navr1.rl.grpo import GRPOTrainer

console = Console()

@click.command()
@click.option("--config", "config_path", type=click.Path(exists=True), default="navr1/configs/default.yaml")
@click.option("--seed", type=int, default=42)
@click.option("--workdir", type=click.Path(), default="./runs/navr1")
def main(config_path: str, seed: int, workdir: str):
    console.rule("[bold green]Nav-R1 Training")
    cfg: NavR1Config = load_config(config_path)
    os.makedirs(workdir, exist_ok=True)

    logger.add(os.path.join(workdir, "train.log"))
    logger.info(f"Loaded config from {config_path}")

    dataloader = build_dataloader(cfg.dataset)
    sim = build_simulator(cfg.simulator)
    policy = FastInSlowPolicy(cfg.model)
    trainer = GRPOTrainer(cfg.rl, policy, sim)

    logger.info("Starting cold-start (supervised) warmup...")
    trainer.cold_start(dataloader)

    logger.info("Starting GRPO reinforcement learning...")
    trainer.train(dataloader)

    console.print("[bold green]Training finished.")

if __name__ == "__main__":
    main()
