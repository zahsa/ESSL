from torch import nn
import torchvision
import torch



class ResNet18_backbone(nn.Module):
    def __init__(self, pretrained=True, seed = 10):
        super().__init__()
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.in_features = resnet.fc.in_features

class largerCNN_backbone(nn.Module):
    def __init__(self, seed = 10):
        super().__init__()
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 64 x 16 x 16
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # output: 128 x 8 x 8
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((4, 4)),  # output: 256 x 4 x 4
            nn.BatchNorm2d(256),
        )
        self.in_features = 4096



class tinyCNN_backbone(nn.Module):
    def __init__(self, seed = 10):
        super().__init__()
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)
        self.backbone = nn.Sequential(
                                nn.Conv2d(3, 6, 5),
                                nn.ReLU(),
                                nn.MaxPool2d(2, 2),
                                nn.Conv2d(6, 16, 5),
                                nn.ReLU(),
                                nn.AdaptiveMaxPool2d((2, 2)),
        )
        self.in_features = 64

