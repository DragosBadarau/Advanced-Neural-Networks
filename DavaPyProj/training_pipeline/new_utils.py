# new_utils.py

import os
import yaml
import argparse
from pydantic import BaseModel, Field, validator
from typing import Optional


# --------- Pydantic Config Models ---------

class OptimizerConfig(BaseModel):
    type: str
    momentum: Optional[float] = 0.9
    nesterov: Optional[bool] = True
    weight_decay: Optional[float] = 0.0


class LRSchedulerConfig(BaseModel):
    type: str
    step_size: Optional[int] = 10
    gamma: Optional[float] = 0.1
    patience: Optional[int] = 10
    factor: Optional[float] = 0.1
    mode: Optional[str] = "min"


class EarlyStoppingConfig(BaseModel):
    enabled: bool = True
    patience: int = 5
    min_delta: float = 0.0
    mode: str = "min"


class LoggingConfig(BaseModel):
    tensorboard: bool = True
    wandb: bool = True
    wandb_project: Optional[str] = "default_project"


class DataAugmentationConfig(BaseModel):
    scheme: str = "none"


class TrainConfig(BaseModel):
    device: str
    dataset: str
    model: str
    batch_size: int
    epochs: int
    learning_rate: float
    cache_data: bool
    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig
    early_stopping: EarlyStoppingConfig
    data_augmentation: DataAugmentationConfig
    logging: LoggingConfig


# --------- Helper Functions ---------

def parse_arguments():
    parser = argparse.ArgumentParser(description="Pydantic Training Config Loader")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    return parser.parse_args()


def load_config(config_path="config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_config() -> TrainConfig:
    args = parse_arguments()
    raw_config = load_config(args.config)

    # Allow environment variable override for device
    raw_config['device'] = os.getenv('DEVICE', raw_config.get('device', 'cuda'))

    return TrainConfig(**raw_config)
