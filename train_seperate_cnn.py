import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from fastDP import PrivacyEngine  # Ensure fastDP is installed and available
import time
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load CIFAR-10 dataset
def get_data_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 normalization
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
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
    
def train_model():
    """
    Train a model with differential privacy by accumulating gradients across an epoch,
    then applying gradient clipping and Gaussian noise addition once after averaging.
    """
    # Load data and model
    trainloader = get_data_loader()  # Assuming this function is defined elsewhere
    model = SimpleCNN().to(device)  # Assuming device is set to 'cuda' or 'cpu' as appropriate

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    # Initialize PrivacyEngine and attach to the optimizer
    privacy_engine = PrivacyEngine(
        model,
        batch_size=256,
        sample_size=len(trainloader.dataset),
        epochs=1,
        target_epsilon=2,
        clipping_fn='automatic',
        clipping_mode='MixOpt',
        origin_params=None,
        clipping_style='all-layer',
        clipping_value=1.0
    )
    privacy_engine.attach(optimizer)

    # Set training configurations
    epochs = 1
    start_train = time.time()
    
    for epoch in range(epochs):
        running_loss = 0.0
        
        # Initialize dictionary to accumulate gradients for each parameter
        accumulated_gradients = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

        # Loop over all batches
        for i, (inputs, labels) in enumerate(trainloader):
            # Move data to device
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            optimizer.zero_grad()  # Zero gradients before the backward pass
            outputs = model(inputs)  # Forward pass

            # Compute loss
            loss = F.cross_entropy(outputs, labels)  # Cross-entropy loss
            loss.backward()  # Backpropagation

            # Accumulate gradients for each parameter
            for name, param in model.named_parameters():
                if param.grad is not None:
                    accumulated_gradients[name] += param.grad

            running_loss += loss.item()

        # After all batches in the epoch, compute the average gradients
        avg_gradients = {name: accumulated_gradients[name] / len(trainloader) for name in accumulated_gradients}

        # Clip averaged gradients and measure time
        start_clip = time.time()
        for name, param in model.named_parameters():
            if param.grad is not None:
                param.grad = avg_gradients[name]  # Replace with averaged gradients
                torch.nn.utils.clip_grad_norm_(param, max_norm=1.0)  # Clip gradients
        end_clip = time.time()
        print(f"Time for gradient clipping: {end_clip - start_clip} seconds")

        # Add Gaussian noise to the averaged gradients and measure time
        noise_std = 1.6  # Define a function to calculate noise standard deviation
        start_noise = time.time()
        for name, param in model.named_parameters():
            if param.grad is not None:
                noise = torch.normal(0, noise_std, size=param.grad.shape).to(param.device)
                param.grad += noise  # Add noise to gradients
        end_noise = time.time()
        print(f"Time for adding Gaussian noise: {end_noise - start_noise} seconds")

        # Step to update model parameters
        optimizer.step()
        optimizer.zero_grad()

        print(f"[{epoch + 1}] epoch loss: {running_loss / len(trainloader):.3f}")

    end_train = time.time()
    print(f"Finished Training, total time = {end_train - start_train} seconds")
    
if __name__ == '__main__':
    train_model()
    
model = SimpleCNN()
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")