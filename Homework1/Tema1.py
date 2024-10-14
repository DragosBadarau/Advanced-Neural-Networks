import torch
import torchvision
import torch.nn.functional
import time


def create_dataloader(dataset, batch_size):
    indices = torch.randperm(len(dataset))
    for i in range(0, len(dataset), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch = [dataset[i][0].view(-1) for i in
                 batch_indices]
        labels = [dataset[i][1] for i in batch_indices]
        yield torch.stack(batch), torch.tensor(labels)


# Activation function: ReLU for hidden layer, Softmax for output layer
def relu(x):
    return torch.clamp(x, min=0)


def softmax(x):
    return torch.exp(x) / torch.sum(torch.exp(x), dim=1, keepdim=True)


def forward(X, W1, b1, W2, b2):
    Z1 = X.mm(W1) + b1
    A1 = relu(Z1)

    Z2 = A1.mm(W2) + b2
    return Z2, A1


def backward(X, A1, Z2, Y, W1, b1, W2, b2, learning_rate, batch_size, device):
    Y_onehot = torch.zeros(Z2.size(0), 10, device=device)
    Y_onehot[range(Z2.size(0)), Y] = 1


    dZ2 = (softmax(
        Z2) - Y_onehot) / batch_size
    dW2 = A1.t().mm(dZ2)
    db2 = torch.sum(dZ2, dim=0)

    dA1 = dZ2.mm(W2.t())
    dZ1 = dA1.clone()
    dZ1[A1 <= 0] = 0
    dW1 = X.t().mm(dZ1)
    db1 = torch.sum(dZ1, dim=0)

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2


# Training the MLP
def train_mlp(device, epochs, learning_rate, batch_size, decay_rate=0.9):
    start_time = time.time()
    input_size = 784
    hidden_size = 100
    output_size = 10

    W1 = torch.randn(input_size, hidden_size, device=device) * 0.001
    b1 = torch.zeros(hidden_size, device=device)
    W2 = torch.randn(hidden_size, output_size, device=device) * 0.001
    b2 = torch.zeros(output_size, device=device)

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                               transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                              transform=torchvision.transforms.ToTensor())

    # Training the MLP
    for epoch in range(epochs):
        current_lr = learning_rate * (decay_rate ** epoch)
        for X_batch, Y_batch in create_dataloader(train_dataset, batch_size):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

            Z2, A1 = forward(X_batch, W1, b1, W2, b2)
            backward(X_batch, A1, Z2, Y_batch, W1, b1, W2, b2, current_lr,
                     batch_size, device)  #

        # Validation after each epoch
        correct, total_loss = 0, 0
        for X_val, Y_val in create_dataloader(test_dataset, batch_size):
            X_val, Y_val = X_val.to(device), Y_val.to(device)

            Z2, _ = forward(X_val, W1, b1, W2, b2)
            total_loss += torch.nn.functional.cross_entropy(Z2, Y_val, reduction='sum').item()  # Calculate loss
            predictions = torch.argmax(softmax(Z2), dim=1)
            correct += torch.sum(predictions == Y_val).item()

        accuracy = correct / len(test_dataset)

    print(f'FINAL Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(test_dataset)}, Accuracy: {accuracy * 100:.2f}%')
    seconds = time.time() - start_time
    print(
        f"Runtime:{seconds // 60} minutes and {seconds % 60:.2f} seconds on {device} with LR ={learning_rate}, batch = {batch_size}, decay= {decay_rate}")


# Hyperparameters
batch_size = 64
learning_rate = 0.1
epochs = 5

# Train the model on GPU (if available)
devices = ['cpu']
if torch.cuda.is_available():
    devices.append('cuda')
    for device in devices:
        # train_mlp(device, epochs, learning_rate, batch_size)
        # train_mlp(device, 7, learning_rate, batch_size)
        # train_mlp(device, 7, 0.01, 64)
        # train_mlp(device, 7, 0.001, 64)
        # train_mlp(device, 5, 0.1, 32, 0.95) # best
        # train_mlp(device, 5, 0.1, 32)
        # train_mlp(device, 5, 0.1, 32, 0.8)
        # train_mlp(device, 7, 0.1, 32, 0.95)
        # train_mlp(device, 7, 0.2, 32, 0.95)
        # train_mlp(device, 7, 0.2, 64, 0.95)
        train_mlp(device, 7, 0.2, 32)
        # train_mlp(device, 5, 0.1, 32, 0.95)
        # train_mlp(device, 7, 0.1, 32, 0.99)
        # train_mlp(device, 7, 0.1, 32) # good
        # train_mlp(device, 7, 0.1, 32, 0.8)
        # train_mlp(device, 7, 0.1, 128)
        # train_mlp(device, 7, 0.01, 32)
        # train_mlp(device, 7, 0.01, 16)
        # train_mlp(device, 7, 0.1, 16)
