import torch
from torch import nn
import copy
from lightly.data import SimCLRCollateFunction, BaseCollateFunction, MultiCropCollateFunction
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead
from lightly.models.modules import SwaVPrototypes
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import BYOLProjectionHead, BYOLPredictionHead
from lightly.models.utils import deactivate_requires_grad
from lightly.models.utils import update_momentum
from lightly.models.modules import NNCLRProjectionHead
from lightly.models.modules import NNCLRPredictionHead
from lightly.models.modules import NNMemoryBankModule

class SimCLR(nn.Module):
    def __init__(self, backbone, seed=10):
        super().__init__()
        # set seeds #
        self.seed = seed
        torch.cuda.manual_seed_all(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.manual_seed(self.seed)

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
            num_workers=12,
        )

        criterion = NTXentLoss().to(device)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.06)
        losses = []
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

class SwaV(nn.Module):
    def __init__(self, backbone, seed=10):
        super().__init__()
        self.seed = seed
        # set seeds #
        torch.cuda.manual_seed_all(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.manual_seed(self.seed)

        self.backbone = backbone.backbone
        self.in_features = backbone.in_features
        self.projection_head = SwaVProjectionHead(self.in_features, 512, 128)
        self.prototypes = SwaVPrototypes(128, n_prototypes=512)
        self.seed = seed


    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        p = self.prototypes(x)
        return p

    def fit(self, transform, dataset, batch_size, num_epochs, device="cuda"):
        self.to(device)
        # employ multi crop collate to custom transform
        # cropping hparams taken directly from Swav collate
        collate_fn = MultiCropCollateFunction( crop_sizes = [24, 8],
                                               crop_counts = [2, 4],
                                               crop_min_scales = [0.14, 0.05],
                                               crop_max_scales = [1.0, 0.14],
                                               transforms=transform)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True,
            num_workers=12,
        )

        criterion = SwaVLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        losses = []
        for epoch in range(num_epochs):
            total_loss = 0
            for batch, _, _ in dataloader:
                self.prototypes.normalize()
                multi_crop_features = [self(x.to(device)) for x in batch]
                high_resolution = multi_crop_features[:2]
                low_resolution = multi_crop_features[2:]
                loss = criterion(high_resolution, low_resolution)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            avg_loss = total_loss / len(dataloader)
            losses.append(avg_loss)
        return losses


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

        self.backbone = backbone.backbone
        self.in_features = backbone.in_features
        self.projection_head = BYOLProjectionHead(self.in_features, 1024, 256)
        self.prediction_head = BYOLProjectionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)


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

    def fit(self, transform, dataset, batch_size, num_epochs, device="cuda"):
        self.to(device)
        collate_fn = BaseCollateFunction(transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True,
            num_workers=12,
        )
        criterion = NegativeCosineSimilarity()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.06)
        losses=[]
        for epoch in range(num_epochs):
            total_loss = 0
            for (x0, x1), _, _ in dataloader:
                update_momentum(self.backbone, self.backbone_momentum, m=0.99)
                update_momentum(self.projection_head, self.projection_head_momentum, m=0.99)
                x0 = x0.to(device)
                x1 = x1.to(device)
                p0 = self(x0)
                z0 = self.forward_momentum(x0)
                p1 = self(x1)
                z1 = self.forward_momentum(x1)
                loss = 0.5 * (criterion(p0, z1) + criterion(p1, z0))
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            avg_loss = total_loss / len(dataloader)
            losses.append(avg_loss)
        return losses

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

        self.backbone = backbone.backbone
        self.in_features = backbone.in_features
        self.projection_head = NNCLRProjectionHead(self.in_features, 512, 128)
        self.prediction_head = NNCLRPredictionHead(128, 512, 128)

    def fit(self, transform, dataset, batch_size, num_epochs, device="cuda"):
        self.to(device)
        memory_bank = NNMemoryBankModule(size=4096)
        memory_bank.to(device)
        collate_fn = BaseCollateFunction(transform)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            drop_last=True,
            num_workers=12,
        )
        criterion = NTXentLoss().to(device)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.06)
        losses = []
        for epoch in range(num_epochs):
            total_loss = 0
            for (x0, x1), _, _ in dataloader:
                x0 = x0.to(device)
                x1 = x1.to(device)
                z0, p0 = self(x0)
                z1, p1 = self(x1)
                z0 = memory_bank(z0, update=False)
                z1 = memory_bank(z1, update=True)
                loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            avg_loss = total_loss / len(dataloader)
            losses.append(float(avg_loss))
        return losses

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


