import torch
import torch.nn as nn

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class VariationalEncoder(nn.Module):
    def __init__(self, height, width, latent_dim):
        super(VariationalEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(height * width, 128),
            nn.ReLU(True)
        )
        self.mean = nn.Linear(128, latent_dim)
        self.log_var = nn.Linear(128, latent_dim)
        
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.encoder(x)
        mean =  self.mean(x)
        log_var = self.log_var(x)
        sigma = torch.exp(log_var)
        epsilon = torch.randn_like(log_var)
        if mean.is_cuda:
            epsilon = epsilon.to(DEVICE)
        z = mean + (sigma * epsilon)
        self.kl_divergence = -0.5 * torch.sum(1 + log_var - mean.square() - sigma)
        return z
    
class Decoder(nn.Module):
    def __init__(self, height, width, latent_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, height * width),
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.decoder(z)
        return z.reshape((-1, 1, 28, 28))
    
class VariationalAutoencoder(nn.Module):
    def __init__(self, height, width, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(height, width, latent_dim)
        self.decoder = Decoder(height, width, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
