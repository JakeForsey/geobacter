from torch import nn
from torchvision import models


class ResNetEmbedding(nn.Module):
    def __init__(self, embedding_dims: int = 2, pretrained: bool = True):
        super().__init__()
        resnet = models.resnet50(pretrained=pretrained)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, embedding_dims)
        )
        self.encoder = resnet

    def forward(self, input):
        return self.encoder(input)


class ResNetTriplet(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, anchor, positive, negative):
        return self.encoder(anchor), self.encoder(positive), self.encoder(negative)
