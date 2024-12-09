import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from fastDP import PrivacyEngine  # Ensure fastDP is installed and available
import time

# Check if CUDA is available and determine GPU count
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Using {num_gpus} GPU(s)")

# 1. Load CIFAR-10 dataset
def get_data_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 normalization
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=4)  # Adjust num_workers for parallel data loading
    return trainloader

# 2. Define a simple CNN model for CIFAR-10
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 3. Training function with DataParallel
def train_model():
    trainloader = get_data_loader()
    model = SimpleCNN()

    # Parallelize model across GPUs
    if num_gpus > 1:
        print("GPU being used.")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Define optimizer and PrivacyEngine
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    privacy_engine = PrivacyEngine(
        model,
        batch_size=256,
        sample_size=len(trainloader.dataset),
        epochs=1,  # Change here to only do one epoch
        target_epsilon=2,
        clipping_fn='automatic',
        clipping_mode='MixOpt',
        origin_params=None,
        clipping_style='all-layer',
        clipping_value=1.0
    )
    privacy_engine.attach(optimizer)

    # Training loop
    epochs = 1  # Only one epoch
    start_train = time.time()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)

            # Compute loss
            loss = F.cross_entropy(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Logging
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0
    end_train = time.time()
    print(f"Finished Training, time = {end_train - start_train}")

# 4. Main entry point
if __name__ == '__main__':
    train_model()
    model = SimpleCNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params}")