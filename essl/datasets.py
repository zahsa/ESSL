import torchvision

class Data:
    def __init__(self):
        self.train_data = None
        self.test_data = None
        self.ssl_data = None
        self.num_classes = None


class Cifar10_transform:
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

class Cifar10(Data):
    def __init__(self, transform=Cifar10_transform()):
        self.train_data = torchvision.datasets.CIFAR10("datasets/cifar10", download=True,
                                                       train = True, transform=transform.train)
        self.test_data = torchvision.datasets.CIFAR10("datasets/cifar10", download=True,
                                                       train=True, transform=transform.test)
        self.ssl_data = torchvision.datasets.CIFAR10("datasets/cifar10", download=True,
                                                       train = False)
        self.num_classes = 10