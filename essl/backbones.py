from torch import nn
import torchvision
import torch

class ResNet18_backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.in_features = resnet.fc.in_features

class tinyCNN_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
                                nn.Conv2d(3, 6, 5),
                                nn.ReLU(),
                                nn.MaxPool2d(2, 2),
                                nn.Conv2d(6, 16, 5),
                                nn.ReLU(),
                                nn.MaxPool2d(2, 2),
        )
        self.in_features = 400

