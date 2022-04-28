from torch import nn
import torchvision

class ResNet18_backbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.in_features = resnet.fc.in_features