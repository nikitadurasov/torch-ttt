"""
TTT for Corrupted MNIST
========================

In this tutorial, we will consider how the original image rotation-based Test-Time Training (TTT) method 
can improve model performance during inference when the data is corrupted by Gaussian noise.

We will use a simple neural network trained on MNIST and add noise to the test set during evaluation.
"""

# %%
# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from datetime import datetime

# %%
# Define a simple neural network for MNIST
activation = nn.ReLU

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu1 = activation()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = activation()
        self.fc3 = nn.Linear(64, 64)
        self.relu3 = activation()
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        return x

# %%
# Prepare the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

# %%
# Instantiate the model, loss function, and optimizer
model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# %%
# Train the model
def train_model(model, train_loader, criterion, optimizer, epochs=5):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()  # Zero the gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)
            loss.backward()  # Backward pass
            optimizer.step()  # Update parameters

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Print training stats
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

    print("Training complete.")

# Train the model
train_model(model, train_loader, criterion, optimizer, epochs=5)

# %%
# Evaluate the model on noisy test data
def evaluate_model(model, test_loader, sigma):
    """
    Evaluate the model on the test set with added Gaussian noise.

    Args:
        model (torch.nn.Module): The trained model.
        test_loader (DataLoader): The DataLoader for the test set.
        sigma (float): The standard deviation of the Gaussian noise.
    """
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    noise = torch.randn([1, 1, 28, 28])  # Fixed Gaussian noise

    for images, labels in test_loader:
        noisy_images = images + noise * sigma
        outputs = model(noisy_images)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}% (with noise sigma={sigma})")

# %%
# Prepare the MNIST test dataset
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# %%
# Evaluate the model
torch.manual_seed(42)
start_time = datetime.now()
evaluate_model(model, test_loader, sigma=10)
print(f"Time taken: {datetime.now() - start_time}")
