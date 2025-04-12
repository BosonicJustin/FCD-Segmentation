import torch
from torch import nn
from transformers import AutoModel


class FCDDetector(nn.Module):
    def __init__(self, encoder='brainIAC', latent_dim=2048):
        super().__init__()

        if encoder == 'brainIAC':
            self.encoder = AutoModel.from_pretrained("Divytak/brainiac")

            # Freeze the encoder parameters
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.projector = nn.Linear(latent_dim, 2)

        # Combine encoder and projector into a single sequential model
        self.classifier = nn.Sequential(
            self.encoder,
            self.projector
        )

        self.loss = nn.BCELoss()  # or any other appropriate loss

    def forward(self, x):
        return self.classifier(x)