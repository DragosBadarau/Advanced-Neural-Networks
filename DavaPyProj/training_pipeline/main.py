import torch
import torch.optim as optim
from torch import nn
from data_loader import get_data_loaders
from model import get_model
from utils import get_config, get_optimizer, get_lr_scheduler, EarlyStopping
from torch.utils.tensorboard import SummaryWriter
import wandb


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


def main():
    config = get_config()
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    # Initialize TensorBoard and wandb
    log_dir = "tensorboard_logs"
    writer = None
    # if config['logging']['tensorboard']:
    #     writer = SummaryWriter(log_dir=log_dir)  # log_dir can be specified if desired

    # if config['logging']['wandb']:
    #     wandb.init(project=config['logging']['wandb_project'], config=config)

    # Set up DataLoaders with data augmentation
    train_loader, test_loader = get_data_loaders(
        config['dataset'],
        config['batch_size'],
        config['cache_data'],
        augmentation_scheme=config["data_augmentation"]["scheme"]
    )

    # Load model based on configuration
    model = get_model(config['model'], config['dataset']).to(device)
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer based on configuration
    optimizer = get_optimizer(config, model.parameters())

    # Initialize learning rate scheduler
    scheduler = get_lr_scheduler(config, optimizer)

    # Initialize Early Stopping
    early_stopping = None
    if config["early_stopping"]["enabled"]:
        early_stopping = EarlyStopping(
            patience=config["early_stopping"]["patience"],
            min_delta=config["early_stopping"]["min_delta"],
            mode=config["early_stopping"]["mode"]
        )

    # Training Loop
    for epoch in range(config['epochs']):
        # Train the model for one epoch
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate the model on the test set
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        # Print metrics
        print(f"Epoch {epoch + 1}/{config['epochs']}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        # Log metrics to TensorBoard
        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/test', test_accuracy, epoch)

        # Log metrics to wandb
        # if config['logging']['wandb']:
        #     wandb.log({
        #         'Loss/train': train_loss,
        #         'Accuracy/train': train_accuracy,
        #         'Loss/test': test_loss,
        #         'Accuracy/test': test_accuracy,
        #         'epoch': epoch
        #     })

        # Step the scheduler if applicable
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

        # Check early stopping
        if early_stopping:
            monitor_value = test_loss if early_stopping.mode == "min" else test_accuracy
            early_stopping(monitor_value)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Close TensorBoard writer outside the loop
    if writer:
        writer.close()

    # # Finish wandb run outside the loop
    # if config['logging']['wandb']:
    #     wandb.finish()


def run_training(config):  # Use config as a parameter instead of loading it inside the function
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')

    # Initialize TensorBoard and wandb
    # log_dir = "tensorboard_logs"
    writer = None
    # if config['logging']['tensorboard']:
    #     writer = SummaryWriter(log_dir=log_dir)  # log_dir can be specified if desired

    # if config['logging']['wandb']:
    #     wandb.init(project=config['logging']['wandb_project'], config=config)

    # Set up DataLoaders with data augmentation
    train_loader, test_loader = get_data_loaders(
        config['dataset'],
        config['batch_size'],
        config['cache_data'],
        augmentation_scheme=config["data_augmentation"]["scheme"]
    )

    # Load model based on configuration
    model = get_model(config['model'], config['dataset']).to(device)
    criterion = nn.CrossEntropyLoss()

    # Initialize optimizer based on configuration
    optimizer = get_optimizer(config, model.parameters())

    # Initialize learning rate scheduler
    scheduler = get_lr_scheduler(config, optimizer)

    # Initialize Early Stopping
    early_stopping = None
    if config["early_stopping"]["enabled"]:
        early_stopping = EarlyStopping(
            patience=config["early_stopping"]["patience"],
            min_delta=config["early_stopping"]["min_delta"],
            mode=config["early_stopping"]["mode"]
        )

    # Training Loop
    for epoch in range(config['epochs']):
        # Train the model for one epoch
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)

        # Evaluate the model on the test set
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)

        # Print metrics
        print(f"Epoch {epoch + 1}/{config['epochs']}, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        # Log metrics to TensorBoard
        # if writer:
        #     writer.add_scalar('Loss/train', train_loss, epoch)
        #     writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        #     writer.add_scalar('Loss/test', test_loss, epoch)
        #     writer.add_scalar('Accuracy/test', test_accuracy, epoch)

        # Log metrics to wandb
        # if config['logging']['wandb']:
        #     wandb.log({
        #         'Loss/train': train_loss,
        #         'Accuracy/train': train_accuracy,
        #         'Loss/test': test_loss,
        #         'Accuracy/test': test_accuracy,
        #         'epoch': epoch
        #     })

        # Step the scheduler if applicable
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

        # Check early stopping
        if early_stopping:
            monitor_value = test_loss if early_stopping.mode == "min" else test_accuracy
            early_stopping(monitor_value)
            if early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

    # Close TensorBoard writer outside the loop
    if writer:
        writer.close()

    # Finish wandb run outside the loop
    # if config['logging']['wandb']:
    #     wandb.finish()


if __name__ == "__main__":
    config = get_config()
    run_training(config)


