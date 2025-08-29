import torch.nn as nn
import torchvision.models as models
from utils import SquarePlus

class KeypointResnetModel(nn.Module):
    def __init__(self, additional_output=False, freeze_resnet = False):
        super(KeypointResnetModel, self).__init__()
        self.conv1 = nn.Conv2d( in_channels=3, out_channels=3, kernel_size=(3, 3), stride=1,
                               padding=1, padding_mode='zeros' )

        self.resnet18 = models.resnet18(weights='DEFAULT')
        self.squareplus = SquarePlus()
        # replacing last layer of resnet
        self.resnet18.fc = nn.Linear(self.resnet18.fc.in_features, 512)

        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(512, 8)
        self.variance = nn.Linear(512, 8)
        self.additional_output = additional_output
        if additional_output:
            self.beta = nn.Linear(512, 8)


    def forward(self, x):
        x = self.conv1(x)
        x = self.resnet18(x)
        x = self.relu(x)
        out = self.squareplus(self.linear1(x))
        var = self.squareplus(self.variance(x))
        if self.additional_output:
            beta = self.squareplus(self.beta(x))
            return out, var, beta
        else:
            return out, var

    def freeze_for_uncertainty(self):
        for param in self.parameters():
            param.requires_grad = False

        for param in self.variance.parameters():
            param.requires_grad = True

        if self.additional_output:
            for param in self.beta.parameters():
                param.requires_grad = True

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True



