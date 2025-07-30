import yaml
import argparse
import os
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler



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
    optimizer_config = config.get("optimizer", {})
    optimizer_type = optimizer_config.get("type", "sgd").lower()  # Default to 'sgd'
    learning_rate = config["learning_rate"]
    weight_decay = optimizer_config.get("weight_decay", 0)

    if optimizer_type == "sgd":
        return optim.SGD(model_params, lr=learning_rate)

    elif optimizer_type == "sgd_momentum":
        momentum = optimizer_config.get("momentum", 0.9)
        return optim.SGD(model_params, lr=learning_rate, momentum=momentum)
def get_optimizer(config, model_params):
    optimizer_config = config.get("optimizer", {})
    optimizer_type = optimizer_config.get("type", "sgd").lower()  # Default to 'sgd'
    learning_rate = config["learning_rate"]
    weight_decay = optimizer_config.get("weight_decay", 0)

    if optimizer_type == "sgd":
        return optim.SGD(model_params, lr=learning_rate)

    elif optimizer_type == "sgd_momentum":
        momentum = optimizer_config.get("momentum", 0.9)
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



def get_lr_scheduler(config, optimizer):
    scheduler_type = config["lr_scheduler"]["type"].lower()

    if scheduler_type == "steplr":
        step_size = config["lr_scheduler"].get("step_size", 10)
        gamma = config["lr_scheduler"].get("gamma", 0.1)
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_type == "reducelronplateau":
        mode = config["lr_scheduler"].get("mode", "min")
        patience = config["lr_scheduler"].get("patience", 10)
        factor = config["lr_scheduler"].get("factor", 0.1)
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, patience=patience, factor=factor)

    elif scheduler_type == "none" or scheduler_type is None:
        return None

    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, mode='min'):
        """
        Args:
            patience (int): How many epochs to wait for an improvement before stopping.
            min_delta (float): Minimum change to qualify as an improvement.
            mode (str): Whether to monitor for a minimum ('min') or maximum ('max') metric.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif self._has_improved(current_score):
            self.best_score = current_score
            self.counter = 0  # reset the counter if we see an improvement
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def _has_improved(self, current_score):
        if self.mode == 'min':
            return current_score < self.best_score - self.min_delta
        elif self.mode == 'max':
            return current_score > self.best_score + self.min_delta
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")