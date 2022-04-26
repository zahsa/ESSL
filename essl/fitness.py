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

import ops
import chromosome


class SimCLR(nn.Module):
    def __init__(self, backbone, out_features):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimCLRProjectionHead(out_features, 512, 128)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)

        z = self.projection_head(x)
        return z

class finetune_model(nn.Module):
    def __init__(self, backbone, in_features, num_outputs):
        super().__init__()
        self.backbone = backbone
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=num_outputs, bias=True),
        )
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.classifier(x)
        return x


def dummy_eval(chromosome):
    """
    dummy evaluation technique, order the augmentations seuentially
    :param chromosome:
    :return:
    """
    permutation = [a[0] for a in chromosome]
    opt = list(range(len(permutation)))
    return sum(np.array(opt) == np.array(permutation))

def gen_augmentation_torch(chromosome: list) -> torchvision.transforms.Compose:
    # gen augmentation
    # dataloader(augmentation)
    aug = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(ops.__dict__[op](intensity=i))
        for op, i in chromosome
    ]+[torchvision.transforms.ToTensor()])
    return aug

def gen_augmentation_PIL(chromosome: list) -> torchvision.transforms.Compose:
    # gen augmentation
    # dataloader(augmentation)
    aug = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(ops.__dict__[op](intensity=i))
        for op, i in chromosome
    ])
    return aug

def ssl_representation(aug: torchvision.transforms.Compose,
                       dataset: LightlyDataset,
                       num_epochs: int,
                       device: str):
    # return features
    # model
    # call ssl training
    # different ssl algorithms
    backbone = torchvision.models.resnet18().to(device)
    model = SimCLR(backbone, out_features=backbone.fc.out_features).to(device)

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
    criterion = NTXentLoss().to(device)
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
    return model


def visualize_chromosomes(save_path, num_samples=5):
    c = chromosome.chromosome_generator()
    cifar10 = torchvision.datasets.CIFAR10("datasets/cifar10", download=True)
    sample = PIL.Image.fromarray(cifar10.data[7])
    sample.save(os.path.join(save_path, "original.jpg"))

    chromosomes = []
    for i in range(num_samples):
        cc = c()
        chromosomes.append(cc)
        aug = gen_augmentation_PIL(cc)
        augmented_im = aug(sample)
        augmented_im.save(os.path.join(save_path, f"{i}.jpg"))
    df = pd.DataFrame(chromosomes, columns=[f"op{i}" for i in range(len(chromosomes[0]))])
    df.to_csv(os.path.join(save_path, "chromosomes.csv"))


def finetune_features(model: torch.nn.Module,
                       train_data: torch.utils.data.Dataset,
                       test_data: torch.utils.data.Dataset,
                       num_epochs: int,
                       device: str,
                       num_outputs: int,
                       batch_size: int = 32):
    """
    COMPONENTS WE MUST SPECIFY FOR FINETUNE:
        - transform pipeline
        - optimizer
        - loss function

    EVALUATION:
        - test acc?
        - test auc?
    :param model:
    :param dataset:
    :param num_epochs:
    :param device:
    :param batch_size:
    :return:
    """
    # finetuning or clustering
    # number of layers, number of neurons
    #
    backbone = model.backbone
    model = finetune_model(backbone, backbone.fc.out_features, num_outputs).to(device)
    trainloader = torch.utils.data.DataLoader(train_data,
                                             batch_size=batch_size)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # train #
    for epoch in range(num_epochs):
        for X, y in trainloader:
            inputs, labels = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    # evaluate #
    testloader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size)
    model.eval()
    y_true = torch.tensor([], dtype=torch.long).to(device)
    pred_probs = torch.tensor([]).to(device)
    # deactivate autograd engine
    with torch.no_grad():
        running_loss = 0.0
        for X, y in testloader:
            inputs = X.to(device)
            labels = y.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            y_true = torch.cat((y_true, labels), 0)
            pred_probs = torch.cat((pred_probs, outputs), 0)

    y_true = y_true.cpu().numpy()
    _, y_pred = torch.max(pred_probs, 1)
    y_pred = y_pred.cpu().numpy()
    # return acc
    return accuracy_score(y_true, y_pred)

def eval_chromosome(chromosome: list):
    aug = gen_augmentation(chromosome)
    features = ssl_representation(aug)
    return finetune_features(features)

class eval_finetune:
    """
    proposed approach:
    wrap above workflow in a classs to store global aspects of the evaluation such as
    dataset and hparams
    """
    def __init__(self,
                 dataset,
                 backbone,
                 ssl_method,
                 ssl_epochs,
                 finetune_opt,
                 finetune_transform,
                 finetune_epochs,
                 finetune_loss):
        pass


if __name__ == "__main__":
    c = chromosome.chromosome_generator()
    cc = c()
    aug = gen_augmentation_torch(cc)
    cifar10 = torchvision.datasets.CIFAR10("datasets/cifar10", download=True)
    dataset = LightlyDataset.from_torch_dataset(cifar10)
    ssl_rep = ssl_representation(aug, dataset, num_epochs=1, device="cuda")

    transform = torchvision.transforms.Compose(
        [
            # torchvision.transforms.ToPILImage(),
            torchvision.transforms.Lambda(lambda image: image.convert('RGB')),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_data = torchvision.datasets.CIFAR10("datasets/cifar10", download=True,
                                           transform=transform, train=True)
    test_data = torchvision.datasets.CIFAR10("datasets/cifar10", download=True,
                                              transform=transform, train=False)


    print("ACC: ", finetune_features(ssl_rep,
                        train_data,
                        test_data,
                        num_epochs=1,
                        device="cuda",
                        num_outputs=10,
                        batch_size=32))



