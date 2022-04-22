import numpy as np
from torch import nn
import torchvision
import torch

from lightly.data import LightlyDataset
from lightly.data import SimCLRCollateFunction, BaseCollateFunction
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead

def dummy_eval(chromosome):
    """
    dummy evaluation technique, order the augmentations seuentially
    :param chromosome:
    :return:
    """
    permutation = [a[0] for a in chromosome]
    opt = list(range(len(permutation)))
    return sum(np.array(opt) == np.array(permutation))

def gen_augmentation(chromosome: list) -> torchvision.transforms.Compose:
    # gen augmentation
    # dataloader(augmentation)
    return



class SimCLR(nn.Module):
    def __init__(self, backbone, in_features):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(in_features, 512, 128)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

class finetune_model(nn.Module):
    def __init__(self, backbone, in_features):
        super().__init__()
        self.backbone = backbone
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.cin_features, out_features=num_outputs, bias=True),
        )



def ssl_representation(aug,dataset, num_epochs, device):
    # return features
    # model
    # call ssl training
    # different ssl algorithms
    backbone = torchvision.models.resnet18()
    model = SimCLR(backbone, in_features=backbone.fc.in_features)

    # custom colate function -> basecollate
    collate_fn = BaseCollateFunction(aug)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=8,
    )
    criterion = NTXentLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.06)
    losses = []
    print("Starting Training")
    for epoch in range(num_epochs):
        total_loss = 0
        for (x0, x1), _, _ in dataloader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            z0 = model(x0)
            z1 = model(x1)
            loss = criterion(z0, z1)
            total_loss += loss.detach()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / len(dataloader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
        losses.append(float(avg_loss))





def finetune_features(model):
    # finetuning or clustering
    # number of layers, number of neurons
    #
    backbone = model.backbone
    return

def eval_chromosome(chromosome: list):
    aug = gen_augmentation(chromosome)
    features = ssl_representation(aug)
    return finetune_features(features)




