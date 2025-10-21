
from typing import List

# Machine learning
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

class Autoencoder(nn.Module):
    """A simple Autoencoder model. Takes a list of dimensions for each layer."""

    def __init__(self, dims: List[int], sigma = nn.ReLU):
        super(Autoencoder, self).__init__()
        encoder_sequences = []
        for i in range(len(dims)-1):
            if dims[i] <= 0:
                raise ValueError("All dimensions must be positive integers.")

            encoder_sequences.append(nn.Linear(dims[i], dims[i+1]))
            encoder_sequences.append(sigma())
        self.encoder = nn.Sequential(
            *encoder_sequences
        )
        
        decoder_sequences = []
        for i in range(len(dims)-1, 0, -1):
            decoder_sequences.append(nn.Linear(dims[i], dims[i-1]))
            decoder_sequences.append(sigma())
        self.decoder = nn.Sequential(
            *decoder_sequences
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten the initial dimensions except the batch dimension
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded