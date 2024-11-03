# sweep_wandb.py
import wandb
from main import run_training
from utils import get_config

# Define sweep configuration
sweep_config = {
    'method': 'grid',  # Could be 'random' if you prefer random search
    'parameters': {
        'model': {
            'values': ['resnet18', 'preactresnet18']
        },
        'data_augmentation': {
            'values': ['none', 'standard', 'advanced']
        },
        'learning_rate': {
            'values': [0.1, 0.075]
        },
        'optimizer_nesterov': {
            'values': [True, False]
        },
        # Add more parameters as needed
    }
}

# Initialize the sweep with Weights & Biases
sweep_id = wandb.sweep(sweep_config, project='pipeline_hw3_sweep')

def sweep_train():
    with wandb.init() as run:
        # Load the base configuration
        config_dict = get_config()  # Loads the default YAML configuration

        # Override the base config with sweep parameters from wandb.config
        config_dict.update({
            'model': wandb.config.model,
            'data_augmentation': {
                'scheme': wandb.config.data_augmentation
            },
            'optimizer': {
                'type': "sgd",
                'learning_rate': wandb.config.learning_rate,
                'momentum': 0.5,
                'nesterov': wandb.config.optimizer_nesterov
            },
            # Include other parameters from wandb.config as needed
        })

        # Ensure logging settings are applied correctly
        config_dict['logging'] = {
            'tensorboard': True,
            'wandb': True,
            'wandb_project': 'pipeline_hw3_sweep',  # Ensure this matches your YAML
        }

        # Run the training function with the updated configuration
        run_training(config_dict)

# Start the sweep agent
wandb.agent(sweep_id, sweep_train)
