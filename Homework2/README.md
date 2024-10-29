# CIFAR-100 Image Classification with VGG16

This project implements a VGG16-based CNN for CIFAR-100 classification, using various optimizer and scheduler strategies to improve accuracy and reduce overfitting.

## Dataset

- **CIFAR-100**: 50,000 training images, 10,000 test images, 100 classes.

## Model Architecture

- VGG16 with batch normalization, adjusted for 100-class output.

## Experiments

| **Experiment**   | **Optimizer**   | **Scheduler**              | **Weight Decay** | **Train Accuracy** | **Val Accuracy** |
|------------------|-----------------|----------------------------|------------------|--------------------|------------------|
| 1. AdamOptimizer | ADAM, LR=0.002  | ReduceLROnPlateau          | 0.01             | 99.7%              | 54.7%            |
| 2. SGDDataAug    | SGD, LR=0.01    | CosineAnnealingLR          | 0.01             | 99.3%              | 59.6%            |
| 3. SGDCycle      | SGD, LR=0.01    | OneCycleLR (Max LR=0.0075) | 0.0025           | 99.9%              | 62%              |
| 4. SGD           | SGD, LR=0.0075  | ReduceLROnPlateau          | 0.01             | 99.9%              | 66.2%            |

## Requirements

- `torch`
- `torchvision`
- `pandas`
- `tqdm`
