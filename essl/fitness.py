import numpy as np
from torch import nn
import torchvision
import torch
import torch.optim as optim
from lightly.data import LightlyDataset
from lightly.data import SimCLRCollateFunction, BaseCollateFunction
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from functools import partial
import PIL
import pandas as pd
import os
from sklearn.metrics import accuracy_score

from essl import ops
from essl import chromosome
from essl import pretext_optimization
from essl import backbones
from essl import losses
from essl import datasets
from essl import evaluate_downstream

def dummy_eval(chromosome):
    """
    dummy evaluation technique, order the augmentations seuentially
    :param chromosome:
    :return:
    """
    permutation = [a[0] for a in chromosome]
    opt = list(range(len(permutation)))
    return sum(np.array(opt) == np.array(permutation))

class pretext_task:
    def __init__(self,
                 method: str,
                 dataset: datasets.Data,
                 backbone: str,
                 num_epochs: int,
                 batch_size: int,
                 device: str):
        self.dataset = LightlyDataset.from_torch_dataset(dataset.ssl_data)
        self.backbone = backbones.__dict__[backbone]
        self.model = pretext_optimization.__dict__[method]
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device



    def __call__(self, transform):
        model = self.model(self.backbone())
        model.fit(transform,
                  self.dataset,
                  self.batch_size,
                  self.num_epochs,
                  self.device)
        return model



class fitness_function:
    """
    proposed approach:
    wrap above workflow in a class to store global aspects of the evaluation such as
    dataset and hparams
    """
    def __init__(self,
                 dataset: str,
                 backbone: str,
                 ssl_task: str,
                 ssl_epochs: int,
                 ssl_batch_size: int,
                 evaluate_downstream_method: str,
                 device: str = "cuda"):
        self.dataset = datasets.__dict__[dataset]()
        self.backbone = backbone
        self.ssl_task = pretext_task(ssl_task,
                                    self.dataset,
                                    self.backbone,
                                    ssl_epochs,
                                    ssl_batch_size,
                                    device
                                    )
        self.evaluate_downstream = evaluate_downstream.__dict__[evaluate_downstream_method](self.dataset)
        self.device = device

    @staticmethod
    def gen_augmentation_torch(chromosome: list) -> torchvision.transforms.Compose:
        # gen augmentation
        transform = torchvision.transforms.Compose([
                                                 torchvision.transforms.Lambda(ops.__dict__[op](intensity=i))
                                                 for op, i in chromosome
                                             ] + [torchvision.transforms.ToTensor()])
        return transform

    def __call__(self, chromosome):
        transform = self.gen_augmentation_torch(chromosome)
        representation = self.ssl_task(transform)
        return self.evaluate_downstream(representation),


if __name__ == "__main__":
    c = chromosome.chromosome_generator()
    cc = c()
    fitness = fitness_function(dataset="Cifar10",
                                 backbone="ResNet18_backbone",
                                 ssl_task="SimCLR",
                                 ssl_epochs=1,
                                 ssl_batch_size=256,
                                 evaluate_downstream_method="finetune",
                                 device="cuda")
    print(fitness(cc))



