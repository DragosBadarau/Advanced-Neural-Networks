import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from data_loader import get_data_loaders
from model import get_model
from utils import get_optimizer, get_lr_scheduler, EarlyStopping

# Import or define functions for your data loaders, model retrieval, etc.
# Example: from your_project import get_data_loaders, get_model, get_optimizer, get_lr_scheduler, train_epoch, evaluate

# 1. Sweep Configuration
sweep_config = {
    'method': 'grid',  # or 'random' for a random search
    'metric': {
        'name': 'test_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'model': {
            'values': ['resnet18', 'preactresnet18']
        },
        'data_augmentation': {
            'values': ['none', 'standard', 'advanced']
        },
        'optimizer': {
            'values': ['adamw', 'sgd']
        },
        'learning_rate': {
            'values': [0.001, 0.01]
        },
        'batch_size': {
            'values': [32, 64]
        }
    }
}

# Initialize the sweep
sweep_id = wandb.sweep(sweep_config, project="cifar100_sweep")


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * inputs.size(0)

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    # Average loss and accuracy over the entire epoch
    epoch_loss = running_loss / total
    epoch_accuracy = 100 * correct / total
    return epoch_loss, epoch_accuracy


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(data_loader.dataset)
    return total_loss / len(data_loader), accuracy


# 2. Training Function

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Set up TensorBoard logging
        writer = SummaryWriter(log_dir="tensorboard_logs")

        # Initialize DataLoaders
        train_loader, test_loader = get_data_loaders(
            dataset="CIFAR100",
            batch_size=config.batch_size,
            cache_data=True,
            augmentation_scheme=config.data_augmentation
        )

        # Initialize Model
        model = get_model(config.model, "CIFAR100").to(device)

        # Initialize Criterion, Optimizer, and Scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = get_optimizer(config, model.parameters())
        scheduler = get_lr_scheduler(config, optimizer)

        # Training Loop
        for epoch in range(10):  # You can also get epochs from config if you want
            train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

            # Log metrics to wandb and TensorBoard
            wandb.log({
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'test_loss': test_loss,
                'test_accuracy': test_accuracy,
                'epoch': epoch
            })

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Accuracy/test', test_accuracy, epoch)

            # Update learning rate scheduler if applicable
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(test_loss)
                else:
                    scheduler.step()

        # Close TensorBoard writer
        writer.close()


# 3. Running the Sweep
wandb.agent(sweep_id, train)
