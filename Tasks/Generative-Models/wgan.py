import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import imageio

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Hyperparameters
EPOCH_SIZE = 200
BATCH_SIZE = 64
LEARNING_RATE = 5e-5
N_CRITIC = 5

dataset_loc = ".../Datasets"
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
to_image = transforms.ToPILImage()

train_dataset = MNIST(dataset_loc, transform=transform, train=True, download=True)
test_dataset  = MNIST(dataset_loc, transform=transform, train=False, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(dataset=test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.generator_layers = nn.Sequential(
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
        x = self.generator_layers(x)
        x = x.view(-1, 1, 28, 28)
        return x

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.critic_layers = nn.Sequential(
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
        )
    def forward(self, x):
        x = x.view(-1, 784)
        return self.critic_layers(x)

# noise z
def noise(n, n_features=100):
    return torch.randn(n, n_features).to(DEVICE)

generator = Generator().to(DEVICE)
critic = Critic().to(DEVICE)

generator_optimizer = torch.optim.RMSprop(generator.parameters(), lr=LEARNING_RATE)
critic_optimizer = torch.optim.RMSprop(critic.parameters(), lr=LEARNING_RATE)

loss_function = nn.BCELoss()

generator.train()
critic.train()
test_noise = noise(100)

for epoch in range(EPOCH_SIZE):
    generator_errors = 0.0
    critic_errors = 0.0
    for batch_index, (images,_) in enumerate(train_loader):
        real_data = images.view(len(images), -1).to(DEVICE)
        fake_data = generator(noise(len(real_data))).detach()
        critic_optimizer.zero_grad()
        critic_error = -torch.mean(critic(real_data)) + torch.mean(critic(fake_data))
        critic_errors -= critic_error
        critic_error.backward()
        critic_optimizer.step()
        for p in critic.parameters():
            p.data.clamp_(-0.01, 0.01)
        
        if (batch_index+1)%N_CRITIC == 0:
            fake_data = generator(noise(len(real_data)))
            generator_optimizer.zero_grad()
            generator_error = -torch.mean(critic(fake_data))
            generator_errors += generator_error
            generator_error.backward()
            generator_optimizer.step()
            
    if (epoch+1)%10 == 0 or epoch == 0:
        print(
            'Epoch {}: Total Generator Losses: {:.7f} Total Critic Losses: {:.7f}'
            .format(
                epoch + 1, 
                generator_errors/ (batch_index*BATCH_SIZE), 
                critic_errors/ (batch_index*BATCH_SIZE)
            )
        )

img = generator(test_noise).cpu().detach()
img = make_grid(img,nrow=10,normalize=True)
imageio.imwrite('hasil tes.png', to_image(img))
