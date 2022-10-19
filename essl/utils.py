import torchvision
import PIL
import pandas as pd
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
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
    d = torch.load(model_path)

    keys_init = model.state_dict().keys()
    for k in keys_init:
        try:
            d[k]
        except KeyError:
            print(k)
    import pdb;
    pdb.set_trace()
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
    X = np.array([[j for j in range(steps)] for i in range(steps)])
    Y = np.array([[i for _ in range(steps)] for i in range(steps)])
    ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    plt.savefig(os.path.join(save_dir, "surface.png"))

def ll_linear_interpolation(model_path1,
                            dataset,
                            backbone,
                            save_dir,
                            steps,
                            model_path2=None
                            ):
    # load models
    backbone = backbones.__dict__[backbone]()
    # model 1
    model1 = finetune_model(copy.deepcopy(backbone.backbone), backbone.in_features, 10)
    model1.load_state_dict(torch.load(model_path1))
    # model 2
    model2 = finetune_model(copy.deepcopy(backbone.backbone), backbone.in_features, 10)
    if model_path2:
        model2.load_state_dict(torch.load(model_path2))
    # get data
    data = datasets.__dict__[dataset]()
    dataloader = torch.utils.data.DataLoader(data.test_data,
                                             batch_size=len(data.test_data), shuffle=False)
    X, y = iter(dataloader).__next__()
    criterion = torch.nn.CrossEntropyLoss()
    metric = loss_landscapes.metrics.Loss(criterion, X, y)

    # compute loss data
    loss_data = loss_landscapes.linear_interpolation(model1, model2, metric, steps, deepcopy_model=True)
    # save loss data
    with open(os.path.join(save_dir, 'linear_interpolation.npy'), 'wb') as f:
        np.save(f, loss_data)
    # plot linear interpolation
    plt.plot([1 / steps * i for i in range(steps)], loss_data)
    plt.savefig(os.path.join(save_dir, "linear_interpolation.png"))
    plt.clf()


if __name__ == "__main__":
    ll_random_plane(model_path="/home/noah/ESSL/final_exps/optimization/exp8_4/4/models/86_downstream.pt",
                    dataset="Cifar10",
                    backbone="largerCNN_backbone",
                    save_dir="/home/noah/ESSL/final_exps/optimization/exp8_4/4/loss_landscapes",
                    distance=10,
                    steps=50)