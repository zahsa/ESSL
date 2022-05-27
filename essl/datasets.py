import torchvision
from torchvision import transforms
from sklearn.model_selection import train_test_split
class Data:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.ssl_data = None
        self.num_classes = None


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

        self.test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

class Cifar10(Data):
    def __init__(self, transform=Cifar10_aug(), split_seed=None):
        self.train_data = torchvision.datasets.CIFAR10("datasets/cifar10", download=True,
                                                       train = True, transform=transform.train)
        if split_seed:
            self.train_data, self.val_data = train_test_split(self.train_data,
                                                               test_size=0.1,
                                                               random_state=split_seed)
        else:
            self.val_data = torchvision.datasets.CIFAR10("datasets/cifar10", download=True,
                                                       train=False, transform=transform.test)

        self.test_data = torchvision.datasets.CIFAR10("datasets/cifar10", download=True,
                                                       train=False, transform=transform.test)
        self.ssl_data = torchvision.datasets.CIFAR10("datasets/cifar10", download=True,
                                                       train = True)
        self.num_classes = 10