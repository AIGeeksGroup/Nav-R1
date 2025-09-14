"""
Configuration utilities for Nav-R1
"""

import yaml
import os
from typing import Dict, Any, Optional
from omegaconf import OmegaConf


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configurations, with override_config taking precedence"""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure"""
    required_keys = [
        "model",
        "dataset", 
        "training",
        "simulator",
        "hardware"
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    return True


def get_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract model configuration"""
    return config.get("model", {})


def get_training_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract training configuration"""
    return config.get("training", {})


def get_dataset_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract dataset configuration"""
    return config.get("dataset", {})


def get_simulator_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract simulator configuration"""
    return config.get("simulator", {})


def get_hardware_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract hardware configuration"""
    return config.get("hardware", {})
