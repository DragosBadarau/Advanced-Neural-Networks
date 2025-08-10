
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm



class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PreActResNet18(nn.Module):
    # Define the PreActResNet18 model as required in Lab 2.
    def __init__(self, num_classes=10):
        super(PreActResNet18, self).__init__()
        # Define layers here
        # Example structure:
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # Add other layers to replicate PreActResNet-18 here...

    def forward(self, x):
        # Define forward pass
        return x



class MLP(nn.Module):
    def __init__(self, input_size=784, num_classes=10):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x



class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2, 2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(2, 2)(x)
        x = x.view(-1, 16 * 5 * 5)
        x = nn.ReLU()(self.fc1(x))
        x = nn.ReLU()(self.fc2(x))
        x = self.fc3(x)
        return x



def get_model(model_name, dataset_name):
    if dataset_name in ["CIFAR10", "CIFAR100"]:
        if model_name == "resnet18":
            return timm.create_model(
                "resnet18", pretrained=True,
                num_classes=(10 if dataset_name == "CIFAR10" else 100)
            )
        elif model_name == "preactresnet18":
            return PreActResNet18(num_classes=(10 if dataset_name == "CIFAR10" else 100))
        else:
            raise ValueError("Model not supported for CIFAR datasets")

    elif dataset_name == "MNIST":
        if model_name == "mlp":
            return MLP(input_size=28 * 28, num_classes=10)
        elif model_name == "lenet":
            return LeNet(num_classes=10)
        else:
            raise ValueError("Model not supported for MNIST")

    else:
        raise ValueError("Dataset not supported. Choose from MNIST, CIFAR10, CIFAR100")
