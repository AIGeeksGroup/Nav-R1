"""
Logging utilities for Nav-R1
"""

import logging
import os
import sys
from typing import Optional
import wandb
from datetime import datetime


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> logging.Logger:
    """Setup logging configuration"""
    
    # Create log directory
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"navr1_{timestamp}.log")
    
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger("navr1")
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger


def get_logger(name: str = "navr1") -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name)


def setup_wandb(
    project: str = "navr1",
    name: Optional[str] = None,
    config: Optional[dict] = None,
    tags: Optional[list] = None
) -> None:
    """Setup Weights & Biases logging"""
    
    if name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"navr1_{timestamp}"
    
    wandb.init(
        project=project,
        name=name,
        config=config,
        tags=tags or [],
    )


def log_metrics(metrics: dict, step: Optional[int] = None, prefix: str = ""):
    """Log metrics to wandb and console"""
    logger = get_logger()
    
    # Log to console
    if prefix:
        metrics_str = {f"{prefix}_{k}": v for k, v in metrics.items()}
    else:
        metrics_str = metrics
    
    logger.info(f"Metrics: {metrics_str}")
    
    # Log to wandb
    if wandb.run is not None:
        wandb.log(metrics_str, step=step)


def log_model_info(model, logger: Optional[logging.Logger] = None):
    """Log model information"""
    if logger is None:
        logger = get_logger()
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")


def log_training_start(config: dict, logger: Optional[logging.Logger] = None):
    """Log training start information"""
    if logger is None:
        logger = get_logger()
    
    logger.info("=" * 60)
    logger.info("Starting Nav-R1 Training")
    logger.info("=" * 60)
    logger.info(f"Configuration: {config}")
    logger.info("=" * 60)


def log_training_end(results: dict, logger: Optional[logging.Logger] = None):
    """Log training end information"""
    if logger is None:
        logger = get_logger()
    
    logger.info("=" * 60)
    logger.info("Training Completed")
    logger.info("=" * 60)
    logger.info(f"Final results: {results}")
    logger.info("=" * 60)
