import torch.nn as nn

class AutoEncoder(nn.Module):

    def __init__(self, archtype ,height, width, latent_dim):
        super().__init__()

        global at, h, w
        at = archtype
        h = height
        w = width
        self.input_dim = height*width
        self.latent_dim = latent_dim
        
        if at == 'MLP':
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, 32),
                nn.ReLU(True),
                nn.Linear(32, 16),
                nn.ReLU(True),
                nn.Linear(16, self.latent_dim)
            )

            self.decoder = nn.Sequential(
                nn.Linear(self.latent_dim, 16),
                nn.ReLU(True),
                nn.Linear(16, 32),
                nn.ReLU(True),
                nn.Linear(32, self.input_dim),
                nn.Sigmoid()
            )
        else:
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=3, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=2, stride=1),
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2),
                nn.ReLU(True),
                nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2),
                nn.Sigmoid(),
            )

    def forward(self, x):
        if at == 'MLP':
            x = x.view(len(x), -1)
            x = self.encoder(x)
            x = self.decoder(x)
            x = x.view(len(x), 1, h, w)
        else:
            x = self.encoder(x)
            x = self.decoder(x)
        return x