import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

root = 'D:\Data for learning AI\MNIST'

transform = transforms.Compose([
    transforms.ToTensor(),                       # đổi ảnh PIL -> tensor (C,H,W), scale [0,1]
    transforms.Normalize((0.5,), (0.5,))         # chuẩn hóa về [-1,1]
])

train_data = datasets.MNIST(root=root, train=True, download=True, transform=transform)
test_data = datasets.MNIST(root=root, train=False, download=True, transform=transform)
#cách chia 1
from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)

'''
#cách chia 2
from torch.utils.data import random_split
train_size = int(0.8 * len(train_data))
val_size = len(train_data) - train_size

train_data, val_data = random_split(train_data, [train_size, val_size])
'''

train_loader = DataLoader(train_data, batch_size=64, shuffle=True) 
val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LeNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

training_loss = []
training_accuracy = []
validation_loss = []
validation_accuracy = []

#evaluate on validation set & test set
def evaluate(model, data_loader):
    model.eval()
    eval_loss = 0.0
    eval_correct = 0
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)
            eval_loss += loss.item()
            _, predicted = torch.max(outputs.data, -1)
            eval_correct += (predicted == labels).sum().item()
    eval_loss /= len(data_loader)
    eval_accuracy = 100 * eval_correct / len(data_loader.dataset)
    return eval_loss, eval_accuracy

#training loop


for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    epoch_correct = 0
    epoch_total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)
        epoch_loss += loss.item()

        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, -1)
        epoch_correct += (predicted == labels).sum().item()
        epoch_total += labels.size(0)
    epoch_loss /= len(train_loader)
    epoch_accuracy = 100 * epoch_correct / epoch_total
    training_loss.append(epoch_loss)
    training_accuracy.append(epoch_accuracy)
    val_loss, val_accuracy = evaluate(model, val_loader)
    validation_loss.append(val_loss)
    validation_accuracy.append(val_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_accuracy:.2f}%, '
          f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')
    

#Save the model
torch.save(model.state_dict(), 'lenet_mnist.pth')

import matplotlib.pyplot as plt
# Plot training & validation loss

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

ax[0].plot(range(1, num_epochs + 1), training_loss, label='Training Loss')
ax[0].plot(range(1, num_epochs + 1), validation_loss, label='Validation Loss')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].set_title('Training and Validation Loss')
ax[0].legend()

# Plot training & validation accuracy
ax[1].plot(range(1, num_epochs + 1), training_accuracy, label='Training Accuracy')
ax[1].plot(range(1, num_epochs + 1), validation_accuracy, label='Validation Accuracy')
ax[1].set_xlabel('Epochs')      
ax[1].set_ylabel('Accuracy (%)')
ax[1].set_title('Training and Validation Accuracy')
ax[1].legend()
plt.show()

# Evaluate on test set

test_loss, test_accuracy = evaluate(model, test_loader)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

