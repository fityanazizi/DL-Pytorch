import time
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

import imageio
from torchvision.utils import make_grid

EPOCH = 200
BATCH_SIZE = 32
LEARNING_RATE = 0.0002
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([transforms.ToTensor()])
to_image = transforms.ToPILImage()
train_data = MNIST('.../Datasets', train=True, download=True, transform=transform)
test_data = MNIST('.../Datasets', train=False, download=True, transform=transform)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x): 
        return self.model(x)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    def forward(self, x): 
        return self.model(x)

def noise(size):
    n = torch.randn(size, 100)
    return n.to(DEVICE)

def discriminator_train_step(real_data, fake_data):
    d_optimizer.zero_grad()
    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real, torch.ones(len(real_data), 1).to(DEVICE))
    error_real.backward()
    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake, torch.zeros(len(fake_data), 1).to(DEVICE))
    error_fake.backward()
    d_optimizer.step()
    return error_real + error_fake

def generator_train_step(fake_data):
    g_optimizer.zero_grad()
    prediction = discriminator(fake_data)
    error = loss(prediction, torch.ones(len(real_data), 1).to(DEVICE))
    error.backward()
    g_optimizer.step()
    return error

discriminator = Discriminator().to(DEVICE)
generator = Generator().to(DEVICE)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE)
loss = nn.BCELoss()

for epoch in range(EPOCH):
    for i, (images, _) in enumerate(train_dataloader):
        real_data = images.view(len(images), -1).to(DEVICE)
        fake_data = generator(noise(len(real_data))).to(DEVICE)
        fake_data = fake_data.detach()
        d_loss = discriminator_train_step(real_data, fake_data)
        fake_data = generator(noise(len(real_data))).to(DEVICE)
        g_loss = generator_train_step(fake_data)
    if epoch == 0 or (epoch+1)%5 == 0:
        print(epoch+1, d_loss.item(), g_loss.item())

z = torch.randn(64, 100).to(DEVICE)
img = generator(z).cpu().detach().view(64, 1, 28, 28)
img = make_grid(img,nrow=10,normalize=True)
imageio.imwrite('hasil tes.png', to_image(img))
