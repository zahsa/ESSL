import torch
from torch import nn
import torchvision

from lightly.data import LightlyDataset
from lightly.data import SimCLRCollateFunction, BaseCollateFunction
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead


class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone.backbone
        self.in_features = backbone.in_features
        self.projection_head = SimCLRProjectionHead(self.in_features, 512, 128)

    def fit(self, transform, dataset, batch_size, num_epochs, device="cuda"):
        self.to(device)
        collate_fn = BaseCollateFunction(transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True,
            num_workers=8,
        )
        criterion = NTXentLoss().to(device)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.06)
        losses = []
        print("Starting Training")
        for epoch in range(num_epochs):
            total_loss = 0
            for (x0, x1), _, _ in dataloader:
                x0 = x0.to(device)
                x1 = x1.to(device)
                z0 = self.forward(x0)
                z1 = self.forward(x1)
                loss = criterion(z0, z1)
                total_loss += loss.detach()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            avg_loss = total_loss / len(dataloader)
            losses.append(float(avg_loss))
        return losses

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z


class BarlowTwins(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = BarlowTwinsProjectionHead(in_features, 2048, 2048)

    def fit(self, transform, dataset, batch_size, num_epochs, device="cuda"):
        raise NotImplementedError

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

class BYOL(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(in_features, 1024, 256)
        self.prediction_head = BYOLProjectionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def fit(self, transform, dataset, batch_size, num_epochs, device="cuda"):
        raise NotImplementedError

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

class MoCo(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = MoCoProjectionHead(in_features, 512, 128)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def fit(self, transform, dataset, batch_size, num_epochs, device="cuda"):
        raise NotImplementedError

    def forward(self, x):
        query = self.backbone(x).flatten(start_dim=1)
        query = self.projection_head(query)
        return query

    def forward_momentum(self, x):
        key = self.backbone_momentum(x).flatten(start_dim=1)
        key = self.projection_head_momentum(key).detach()
        return key

class NNCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = NNCLRProjectionHead(in_features, 512, 128)
        self.prediction_head = NNCLRPredictionHead(128, 512, 128)

    def fit(self, transform, dataset, batch_size, num_epochs, device="cuda"):
        raise NotImplementedError

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p



class SimSiam(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(in_features, 512, 128)
        self.prediction_head = SimSiamPredictionHead(128, 64, 128)

    def fit(self, transform, dataset, batch_size, num_epochs, device="cuda"):
        raise NotImplementedError

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p


