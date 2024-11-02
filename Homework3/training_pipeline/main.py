import torch
import torch.optim as optim
from torch import nn
from data_loader import get_data_loaders
from model import get_model
from utils import get_config, get_optimizer


def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


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

    # Set up DataLoaders
    train_loader, test_loader = get_data_loaders(config['dataset'], config['batch_size'], config['cache_data'])

    # Load model based on configuration
    model = get_model(config['model'], config['dataset']).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    # Initialize optimizer based on configuration
    optimizer = get_optimizer(config, model.parameters())

    # Training Loop
    for epoch in range(config['epochs']):
        train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        print(f"Epoch {epoch + 1}/{config['epochs']}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")


if __name__ == "__main__":
    main()
