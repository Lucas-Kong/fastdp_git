import time
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fastDP import PrivacyEngine
from transformers import ViTForImageClassification

# Hyperparameters
batch_size = 256
learning_rate = 0.05
epochs = 3
target_epsilon = 2.0

# Data loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Model setup
model = ViTForImageClassification.from_pretrained('google/vit-large-patch16-224', num_labels=10)

# Optimizer
optimizer = SGD(model.parameters(), lr=learning_rate)

# Privacy Engine
privacy_engine = PrivacyEngine(
    model,
    batch_size=batch_size,
    sample_size=len(train_dataset),
    epochs=epochs,
    target_epsilon=target_epsilon,
    clipping_fn='automatic',
    clipping_mode='MixOpt',
    origin_params=None,
    clipping_style='all-layer',
)

# Attaching the privacy engine to the optimizer
privacy_engine.attach(optimizer)

# Measure time for a single training iteration
model.train()
data, target = next(iter(train_loader))  # Get a single batch
start_time = time.time()

# Training step
optimizer.zero_grad()
output = model(data).logits
loss = F.cross_entropy(output, target)
loss.backward()
optimizer.step()

end_time = time.time()
time_per_iteration = end_time - start_time
print(f"Time consumed for a single training iteration: {time_per_iteration:.6f} seconds")