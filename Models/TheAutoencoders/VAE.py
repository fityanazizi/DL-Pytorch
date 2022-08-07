import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim) -> None:
        super().__init__()

class Decoder(nn.Module):
    def __init__(self, latent_dim) -> None:
        super().__init__()

class VariationalAutoEncoder(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super().__init__()