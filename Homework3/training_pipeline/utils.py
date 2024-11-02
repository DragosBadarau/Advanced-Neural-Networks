import yaml
import argparse
import os
import torch.optim as optim

def load_config(default_path="config.yaml"):
    with open(default_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch Training Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    return parser.parse_args()

def get_config():
    args = parse_arguments()
    config = load_config(args.config)
    # Allow environment variables to override config
    config['device'] = os.getenv('DEVICE', config.get('device', 'cuda'))
    return config


def get_optimizer(config, model_params):
    optimizer_type = config["optimizer"]["type"].lower()
    learning_rate = config["learning_rate"]
    weight_decay = config["optimizer"].get("weight_decay", 0)

    if optimizer_type == "sgd":
        return optim.SGD(model_params, lr=learning_rate)

    elif optimizer_type == "sgd_momentum":
        momentum = config["optimizer"].get("momentum", 0.9)
        return optim.SGD(model_params, lr=learning_rate, momentum=momentum)

    elif optimizer_type == "sgd_nesterov":
        momentum = config["optimizer"].get("momentum", 0.9)
        nesterov = config["optimizer"].get("nesterov", True)
        return optim.SGD(model_params, lr=learning_rate, momentum=momentum, nesterov=nesterov)

    elif optimizer_type == "sgd_weight_decay":
        momentum = config["optimizer"].get("momentum", 0.9)
        return optim.SGD(model_params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    elif optimizer_type == "adam":
        return optim.Adam(model_params, lr=learning_rate)

    elif optimizer_type == "adamw":
        return optim.AdamW(model_params, lr=learning_rate, weight_decay=weight_decay)

    elif optimizer_type == "rmsprop":
        momentum = config["optimizer"].get("momentum", 0.9)
        return optim.RMSprop(model_params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")