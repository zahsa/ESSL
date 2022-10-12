import torchvision
import PIL
import pandas as pd
import os
import torch
import numpy as np
from essl import chromosome
from essl import backbones
from essl.evaluate_downstream import finetune_model
from essl import datasets

import loss_landscapes

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

def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismatch found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')
    else:
        print('different models')

def compare_dicts(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.items(), model_2.items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


def ll_random_plane(model_path,
                      dataset,
                      backbone,
                      save_dir,
                      distance,
                      steps):
    # load model
    backbone = backbones.__dict__[backbone]()
    model = finetune_model(backbone.backbone, backbone.in_features, 10)
    model.load_state_dict(torch.load(model_path))

    # get data
    data = datasets.__dict__[dataset]()
    dataloader = torch.utils.data.DataLoader(data.test_data,
                                              batch_size=len(data.test_data), shuffle=False)
    X, y = iter(dataloader).__next__()
    criterion = torch.nn.CrossEntropyLoss()
    metric = loss_landscapes.metrics.Loss(criterion, X, y)

    # compute random plane
    loss_data_fin = loss_landscapes.random_plane(model, metric, distance, steps, normalization='filter', deepcopy_model=True)
    # save loss data
    with open(os.path.join(save_dir, 'random_plane.npy'), 'wb') as f:
        np.save(f, loss_data_fin)

    # plot contour
    plt.contour(loss_data_fin, levels=50)
    plt.savefig(os.path.join(save_dir, "contour.png"))
    plt.clf()

    # plot surface
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
    Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
    ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    plt.savefig(os.path.join(save_dir, "surface.png"))





