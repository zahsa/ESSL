import torchvision
import PIL
import pandas as pd
import os

from essl import chromosome

def gen_augmentation_PIL(chromosome: list) -> torchvision.transforms.Compose:
    # gen augmentation
    # dataloader(augmentation)
    aug = torchvision.transforms.Compose([
        torchvision.transforms.Lambda(ops.__dict__[op](intensity=i))
        for op, i in chromosome
    ])
    return aug

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

def id_generator() -> int:
    """
    simple generator to generate undefined number of unique,
    sequential ids
    :yield: unique id
    """
    id = -1
    while True:
        id += 1
        yield id