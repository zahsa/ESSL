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
from essl import pretext_selection
from essl import backbones
from essl import losses
from essl import datasets
from essl import evaluate_downstream
import time

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
                 device: str,
                 seed: int=10
                 ):
        self.seed = seed
        self.dataset = LightlyDataset.from_torch_dataset(dataset.ssl_data)
        self.backbone = backbones.__dict__[backbone]
        self.model = pretext_selection.__dict__[method]
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.device = device



    def __call__(self, transform):
        model = self.model(self.backbone(self.seed))
        loss = model.fit(transform,
                  self.dataset,
                  self.batch_size,
                  self.num_epochs,
                  self.device)
        return model, loss




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
                 evaluate_downstream_kwargs: dict = {},
                 device: str = "cuda",
                 seed: int=10):

        # set seeds #
        self.seed = seed
        torch.cuda.manual_seed_all(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.manual_seed(self.seed)

        self.dataset = datasets.__dict__[dataset](seed=seed)
        self.backbone = backbone
        self.ssl_task = pretext_task(method=ssl_task,
                                    dataset=self.dataset,
                                    backbone=self.backbone,
                                    num_epochs=ssl_epochs,
                                    batch_size=ssl_batch_size,
                                    device=device,
                                    seed=self.seed
                                    )
        self.evaluate_downstream = evaluate_downstream.__dict__[evaluate_downstream_method](dataset=self.dataset,
                                                                                             seed=seed,
                                                                                            **evaluate_downstream_kwargs)
        self.downstream_losses = {}
        self.device = device



    @staticmethod
    def gen_augmentation_torch(chromosome: list) -> torchvision.transforms.Compose:
        # gen augmentation
        transform = torchvision.transforms.Compose([
                                                 torchvision.transforms.Lambda(ops.__dict__[op](intensity=i))
                                                 for op, i in chromosome
                                             ] + [torchvision.transforms.ToTensor()])
        return transform

    def clear_downstream_losses(self):
        self.downstream_losses = {}

    def __call__(self, chromosome, return_losses=False):
        t1 = time.time()
        transform = self.gen_augmentation_torch(chromosome)
        representation, ssl_losses = self.ssl_task(transform)
        train_losses, train_accs, val_losses, val_accs, test_acc = self.evaluate_downstream(representation, report_all_metrics=True)
        print("time to eval: ", time.time() - t1)
        if return_losses:
            return ssl_losses, train_losses, train_accs, val_losses, val_accs, test_acc
        else:
            # store the losses with id of chromosome
            self.downstream_losses[chromosome.id] = train_losses
            return test_acc,


if __name__ == "__main__":
    c = chromosome.chromosome_generator()
    cc = c()
    print("seed: ", torch.seed())
    fitness = fitness_function(dataset="Cifar10",
                                 backbone="largerCNN_backbone",
                                 ssl_task="SwaV",
                                 ssl_epochs=1,
                                 ssl_batch_size=64,
                                 evaluate_downstream_method="finetune",
                                 device="cuda")
    print("seed: ", torch.cuda.seed())
    print(fitness(cc, return_losses=True))
    print("seed: ", torch.cuda.seed())
    import pdb;
    pdb.set_trace()



