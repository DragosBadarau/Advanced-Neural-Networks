import yaml
import argparse
import os

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
