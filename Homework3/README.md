# Python implementation of a configurable, device-agnostic training pipeline in PyTorch.

Configurable either trough
command line arguments, configuration files or environment variables.

### Key Features Implemented

1. **Device Agnostic Training**: The pipeline supports both CPU and GPU.
2. **Configurable Dataset Support**: Supports training on MNIST, CIFAR-10, and CIFAR-100.
3. **Efficient Data Caching**: Datasets include caching to improve loading times.
4. **Flexible DataLoaders**: Configurable data loaders for training and testing phases.
5. **Model Selection**: Configurable models; included support for `resnet18`, `preactresnet18` (for CIFAR), and `MLP`, `LeNet` (for MNIST).
6. **Optimizer Selection**: Supports SGD, Adam, AdamW, RMSprop, etc.
7. **Learning Rate Schedulers**: Integrated with `StepLR` and `ReduceLROnPlateau`.
8. **Early Stopping**: Configurable early stopping to avoid overfitting.
9. **Data Augmentation Schemes**: Allows choosing between different augmentation strategies.
10. **Metrics Reporting**: Integrated with `wandb` and TensorBoard for metrics reporting.
11. **Parameter Sweep**: Conducted a parameter sweep on CIFAR-100, exploring model architecture, data augmentation, and optimizer settings.

To execute the script : 
: **python main.py --config config.yaml**

### Wandb Report with multiple tests taken 

https://api.wandb.ai/links/dragos_badarau-alexandru-ioan-cuza-university-of-iasi/vqs5cmzq 

![Statistics Image](training_pipeline/pictures/Wandb%20Stats.png)


## Parameter Sweep Setup

### Parameters Varied

For the sweep, I experimented with the following parameters in the `wandb_sweep.py` file:

- **Model Architecture**: `resnet18` and `preactresnet18`
- **Data Augmentation**: None, Standard and Advanced schemes
- **Optimizer**: SGD
- **Learning Rate**:  [0.1, 0.075]
- **Nesterov**: True or False 

### Hyperparameter Configurations

Each run used CIFAR-100 as the dataset, with `ReduceLROnPlateau` as the learning rate scheduler.

To execute the script : 
: **python wandb_sweep.py**


### Expected points to receive: 14 