import time
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

EPOCH = 25
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([transforms.ToTensor()])
train_data = MNIST('.../Datasets', train=True, download=True, transform=transform)
test_data = MNIST('.../Datasets', train=False, download=True, transform=transform)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

class ResidualBlock(nn.Module):
    def __init__(self, input_channels, num_channels, identity=False) -> None:
        super().__init__()
        if identity:
            stride = 2
            self.conv3 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=1, stride=2)
        else:
            stride = 1
            self.conv3 = None

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(num_features=num_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(num_features=num_channels),
        )
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.conv1(x)
        x = self.conv2(x)
        if self.conv3:
            shortcut = self.conv3(shortcut)
        x += shortcut
        return self.relu(x)

class ResNet18(nn.Module):
    def __init__(self, input_channel, ResidualBlock, num_output) -> None:
        super().__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = nn.Sequential(
            ResidualBlock(input_channels=64, num_channels=64),
            ResidualBlock(input_channels=64, num_channels=64)
        )
        self.layer2 = nn.Sequential(
            ResidualBlock(input_channels=64, num_channels=128, identity=True),
            ResidualBlock(input_channels=128, num_channels=128)
        )
        self.layer3 = nn.Sequential(
            ResidualBlock(input_channels=128, num_channels=256, identity=True),
            ResidualBlock(input_channels=256, num_channels=256)
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(input_channels=256, num_channels=512, identity=True),
            ResidualBlock(input_channels=512, num_channels=512)
        )
        self.layer5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, num_output)
            # softmax?
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.layer5(x)

resnet18 = ResNet18(1, ResidualBlock, 10)
resnet18.to(DEVICE)
resnet18.train()
loss_function = nn.CrossEntropyLoss()
optimizer= torch.optim.Adam(resnet18.parameters(), lr = LEARNING_RATE)
now = time.time()
for epoch in range(EPOCH):
    overall_loss = 0
    for batch_index, (x, y) in enumerate(train_dataloader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        output = resnet18(x)
        loss = loss_function(output, y)
        overall_loss += loss.item()
        loss.backward()
        optimizer.step()
    if epoch == 0 or (epoch+1)%5 == 0:
        print(epoch+1, overall_loss / (batch_index*BATCH_SIZE))
print(time.time() - now)
