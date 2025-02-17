import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from model import CNN
from IPython.display import display, clear_output

class CustomMNIST(Dataset):
    def __init__(self, images_file, labels_file, transform=None):
        try:
            with open(images_file, 'rb') as f:
                self.images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
            with open(labels_file, 'rb') as f:
                self.labels = np.frombuffer(f.read(), np.uint8, offset=8)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            raise

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Load MNIST dataset
transform = transforms.ToTensor()

# Construct absolute paths
mnist_dir = '/Users/arnav/Documents/MNIST DATA RAW'
train_images_path = os.path.join(mnist_dir, 'train-images.idx3-ubyte')
train_labels_path = os.path.join(mnist_dir, 'train-labels.idx1-ubyte')
test_images_path = os.path.join(mnist_dir, 't10k-images.idx3-ubyte')
test_labels_path = os.path.join(mnist_dir, 't10k-labels.idx1-ubyte')

train_data = CustomMNIST(
    train_images_path,
    train_labels_path,
    transform=transform
)
test_data = CustomMNIST(
   test_images_path,
   test_labels_path,
    transform=transform
)

# Split training data into training and validation sets
train_size = int(0.9 * len(train_data))
val_size = len(train_data) - train_size
train_dataset, val_dataset = random_split(train_data, [train_size, val_size])

# Data loaders
loaders = {
    'train': DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=0),
    'val': DataLoader(val_dataset, batch_size=100, shuffle=True, num_workers=0),
    'test': DataLoader(test_data, batch_size=100, shuffle=True, num_workers=0)
}

# Initialize model, optimizer, and loss function
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
loss_fn = nn.CrossEntropyLoss()

train_losses, val_losses = [], []
train_accs, val_accs = [], []

def train(epoch):
    model.train()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(loaders['train']):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % 20 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(loaders["train"].dataset)} ({100. * batch_idx / len(loaders["train"]):.0f}%)]\tLoss: {loss.item():.6f}')

    train_accuracy = 100 * correct / total
    train_accs.append(train_accuracy)
    print(f'Epoch {epoch}: Training Accuracy: {train_accuracy:.2f}%')

def validate():
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loaders['val']:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += loss_fn(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    val_loss /= len(loaders['val'])
    val_losses.append(val_loss)
    val_accuracy = 100. * correct / total
    val_accs.append(val_accuracy)
    print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%')
    return val_loss

def main():
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    n_epochs = 10
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        val_loss = validate()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping")
                break

if __name__ == "__main__":
    main()


