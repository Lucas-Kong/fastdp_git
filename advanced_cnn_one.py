import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from fastDP import PrivacyEngine
import time
import multiprocessing
import config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data augmentation with explicit resizing
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Ensure all images are resized to 32x32
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Ensure test images are resized to 32x32
    transforms.ToTensor(),
])


# Custom Dataset class
class CIFAR10_dataset(Dataset):
    def __init__(self, partition="train", transform=None):
        print("\nLoading CIFAR10", partition, "Dataset...")
        self.partition = partition
        self.transform = transform
        if self.partition == "train":
            self.data = torchvision.datasets.CIFAR10(root = './data', train=True, download=True)
        else:
            self.data = torchvision.datasets.CIFAR10(root = './data', train=False, download=True)
        print("\tTotal Len.:", len(self.data), "\n", 50 * "-")

    def from_pil_to_tensor(self, image):
        return torchvision.transforms.ToTensor()(image)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Image
        image = self.data[idx][0]
        image_tensor = self.transform(image)

        # Label
        label = torch.tensor(self.data[idx][1])
        label = F.one_hot(label, num_classes=10).float()  # One-hot encoding of the label

        return {"img": image_tensor, "label": label}

# Load datasets with augmentation for training and no augmentation for testing
train_dataset = CIFAR10_dataset(partition="train", transform=train_transform)
test_dataset = CIFAR10_dataset(partition="test", transform=test_transform)

# Set batch size and number of workers for data loading
batch_size = 100
num_workers = multiprocessing.cpu_count() - 1
print("Num workers:", num_workers)

train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
test_dataloader = DataLoader(test_dataset, batch_size, shuffle=False, num_workers=num_workers)

# Define Advanced CNN Model (already provided)
class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(AdvancedCNN, self).__init__()
        # Define the layers (same as in previous code)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32 * 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32 * 2, out_channels=64 * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64 * 2, out_channels=64 * 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64 * 2, out_channels=128 * 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=128 * 2, out_channels=128 * 2, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=128 * 2, out_channels=128 * 2, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=128 * 2, out_channels=256 * 2, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(in_channels=256 * 2, out_channels=256 * 2, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=256 * 2, out_channels=256 * 2, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(32 * 2)
        self.bn2 = nn.BatchNorm2d(128 * 2)
        self.bn3 = nn.BatchNorm2d(256 * 2)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(0.2)

        self.fc1 = nn.Linear(4096 * 2, 4096 * 2)
        self.fc2 = nn.Linear(4096 * 2, 2048 * 2)
        self.fc3 = nn.Linear(2048 * 2, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.maxpool(x)

        x = self.relu(self.bn2(self.conv4(x)))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = self.relu(self.bn3(self.conv7(x)))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.maxpool(x)
        x = self.dropout(x)

        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# Training function with DP-SGD and augmented dataset
def train_model():
    # Train model on augmented data
    trainloader = train_dataloader
    model = AdvancedCNN()
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    
    # Initialize PrivacyEngine
    from decimal import Decimal
    config.global_gaussian = Decimal(0)
    config.global_clipping = Decimal(0)
    config.count_gaussian = 0
    config.count_clipping = 0
    config.count_total_gaussian = 0
    config.count_total_clipping = 0

    # Define optimizer
    optimizer = optim.SGD(model.parameters(), lr=0.05)

    # PrivacyEngine
    privacy_engine = PrivacyEngine(
        model,
        batch_size=256,
        sample_size=len(trainloader.dataset),
        epochs=1,  # Only one epoch
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
        for i, batch in enumerate(trainloader):
            inputs, labels = batch['img'].to(device), batch['label'].to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Loss function (cross-entropy)
            loss = F.cross_entropy(outputs, labels)

            # Backward pass
            loss.backward()
            
            # PrivacyEngine Gradient Clipping and Adding Gaussian noise
            optimizer.step()

            # Logging
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

    end_train = time.time()
    print(f"Finished Training, time = {end_train - start_train}")
    print(f"time_gaussian: {config.global_gaussian}, time_clipping: {config.global_clipping}")
    print(f"count_gaussian: {config.count_gaussian}, count_clipping: {config.count_clipping}")
    print(f"count_total_gaussian: {config.count_total_gaussian}, count_total_clipping: {config.count_total_clipping}")

# Call the training function
if __name__ == '__main__':
    train_model()
