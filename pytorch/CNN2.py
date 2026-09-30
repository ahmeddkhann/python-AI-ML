import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


train_data = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

test_data = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    transform=transform,
    download=True
)


train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False,
    num_workers=0
)


class EfficientConvulationNeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
           in_channels= 64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(
           in_channels= 128,
           out_channels=256,
            kernel_size=3,
            padding=1
        )

        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(
            in_features=256,
            out_features=10
        )

    def forward(self, x):

        x = self.pool(F.relu(self.bn1(self.conv1(x))))

        x = self.pool(F.relu(self.bn2(self.conv2(x))))

        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = F.relu(self.bn4(self.conv4(x)))

        x = self.global_pool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


net = EfficientConvulationNeuralNetwork()

loss_function = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    net.parameters(),
    lr=0.001
)

epochs = 50


for epoch in range(epochs):

    net.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:

        optimizer.zero_grad()

        outputs = net(inputs)

        loss = loss_function(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

    train_loss = running_loss / len(train_loader)

    train_accuracy = 100 * correct / total


    net.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for inputs, labels in test_loader:

            outputs = net(inputs)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    test_accuracy = 100 * correct / total

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {train_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.2f}% "
        f"Test Accuracy: {test_accuracy:.2f}%"
    )


torch.save(
    net.state_dict(),
    "cnn3.pth"
)