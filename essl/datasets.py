import torchvision
from torchvision import transforms
import torch

from torch.utils.data import random_split
class Data:
    def __init__(self, seed=10):
        self.train_data = None
        self.test_data = None
        self.ssl_data = None
        self.num_classes = None
        # set seeds #
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

class DatasetFromSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)
#
# class DataSet:
#     def __init__(self):

class Cifar10_noaug:
    def __init__(self):
        self.train = torchvision.transforms.Compose(
            [
                torchvision.transforms.Lambda(lambda image: image.convert('RGB')),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        self.test = torchvision.transforms.Compose(
            [
                torchvision.transforms.Lambda(lambda image: image.convert('RGB')),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((32, 32)),
                torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

class Cifar10_aug:
    def __init__(self):
        self.train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.val = transforms.Compose([
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        self.test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

class Cifar10(Data):
    def __init__(self, transform=Cifar10_aug(), seed=10):
        super().__init__(seed=seed)
        self.train_data = torchvision.datasets.CIFAR10("datasets/cifar10", download=True,
                                                       train = True, transform=transform.train)
        if seed:
            self.train_data, self.val_data = random_split(self.train_data,
                                                               [int(len(self.train_data)*0.9), len(self.train_data) - int(len(self.train_data)*0.9)],
                                                               generator=torch.Generator().manual_seed(seed))



        else:
            self.val_data = torchvision.datasets.CIFAR10("datasets/cifar10", download=True,
                                                       train=False, transform=transform.test)

        self.test_data = torchvision.datasets.CIFAR10("datasets/cifar10", download=True,
                                                       train=False, transform=transform.test)
        self.ssl_data = torchvision.datasets.CIFAR10("datasets/cifar10", download=True,
                                                       train = True)
        self.num_classes = 10


class SVHN_aug:
    def __init__(self):
        self.train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728),
                                         (0.1980, 0.2010, 0.1970)),
        ])
        self.val = transforms.Compose([
            transforms.Normalize((0.4377, 0.4438, 0.4728),
                                         (0.1980, 0.2010, 0.1970)),
        ])
        self.test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728),
                                         (0.1980, 0.2010, 0.1970)),
        ])

class SVHN(Data):
    def __init__(self, transform=SVHN_aug(), seed=10):
        super().__init__(seed=seed)
        self.train_data = torchvision.datasets.SVHN("datasets/SVHN", download=True,
                                                       split="train", transform=transform.train)
        if seed:
            self.train_data, self.val_data = random_split(self.train_data,
                                                               [int(len(self.train_data)*0.9), len(self.train_data) - int(len(self.train_data)*0.9)],
                                                               generator=torch.Generator().manual_seed(seed))



        else:
            self.val_data = torchvision.datasets.SVHN("datasets/SVHN", download=True,
                                                       split="test", transform=transform.test)

        self.test_data = torchvision.datasets.SVHN("datasets/SVHN", download=True,
                                                       split="test", transform=transform.test)
        self.ssl_data = torchvision.datasets.SVHN("datasets/SVHN", download=True,
                                                       split="train")
        self.num_classes = 10