import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os


def get_data_transforms():
    # Define transformations for data augmentation
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    return train_transform, test_transform


def get_dataset(dataset_name, train=True, transform=None):
    # Set up datasets with cache
    data_dir = "./data_cache"
    os.makedirs(data_dir, exist_ok=True)

    if dataset_name == "MNIST":
        return datasets.MNIST(root=data_dir, train=train, transform=transform, download=True)
    elif dataset_name == "CIFAR10":
        return datasets.CIFAR10(root=data_dir, train=train, transform=transform, download=True)
    elif dataset_name == "CIFAR100":
        return datasets.CIFAR100(root=data_dir, train=train, transform=transform, download=True)
    else:
        raise ValueError("Unsupported dataset. Choose from MNIST, CIFAR10, CIFAR100.")


def get_data_loaders(dataset_name, batch_size, cache_data):
    train_transform, test_transform = get_data_transforms()

    train_dataset = get_dataset(dataset_name, train=True, transform=train_transform)
    test_dataset = get_dataset(dataset_name, train=False, transform=test_transform)

    # Configure DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
