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


def get_data_loaders(dataset_name, batch_size, cache_data, augmentation_scheme="none"):
    # Get appropriate transformation for training and testing datasets
    train_transform = get_augmentation_transforms(augmentation_scheme)

    # Test set: No augmentation, just normalization
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)) if dataset_name == "MNIST" else transforms.Normalize((0.5, 0.5, 0.5),
                                                                                                  (0.5, 0.5, 0.5)),
    ])

    # Choose the dataset
    if dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=test_transform)
    elif dataset_name == "CIFAR100":
        train_dataset = datasets.CIFAR100(root="data", train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root="data", train=False, download=True, transform=test_transform)
    elif dataset_name == "MNIST":
        train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=train_transform)
        test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last= True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last= False)

    return train_loader, test_loader


from torchvision import datasets, transforms


def get_augmentation_transforms(scheme):
    CIFAR100_TRAIN_MEAN = (0.5071, 0.4867, 0.4408)
    CIFAR100_TRAIN_STD = (0.2675, 0.2565, 0.2761)
    if scheme == "none":
        # No augmentation, only normalization
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN,
                                 CIFAR100_TRAIN_STD) if datasets == "MNIST" else transforms.Normalize(CIFAR100_TRAIN_MEAN,
                                                                                                  CIFAR100_TRAIN_STD),
        ])

    elif scheme == "standard":
        # Standard augmentations
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            # transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN,
                                 CIFAR100_TRAIN_STD)
        ])

    elif scheme == "advanced":
        # Advanced augmentations (additional color jitter and random erasing)
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_TRAIN_MEAN,
                                 CIFAR100_TRAIN_STD),
            transforms.RandomErasing(scale=(0.02, 0.2))
        ])

    else:
        raise ValueError(f"Unsupported augmentation scheme: {scheme}")

    return transform
