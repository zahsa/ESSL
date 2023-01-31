import pdb

import torchvision
import PIL
import pandas as pd
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import click
import copy
import json


from torch.nn.utils import (
  parameters_to_vector as Params2Vec,
  vector_to_parameters as Vec2Params
)
from essl.backbones import largerCNN_backbone
from essl.datasets import Cifar10

from essl import chromosome
from essl import backbones
from essl.evaluate_downstream import finetune_model
from essl.fitness import fitness_function
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

def visualize_random_chromosomes(save_path, num_samples=5):
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


def visualize_chromosomes(save_path, chromosomes):
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
    X = np.array([[j for j in range(steps)] for i in range(steps)])
    Y = np.array([[i for _ in range(steps)] for i in range(steps)])
    ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    plt.savefig(os.path.join(save_dir, "surface.png"))

@click.command()
@click.option("--model_path", type=str, help="path to model")
@click.option("--dataset", type=str, help="dataset to be used")
@click.option("--backbone", type=str, help="backbone to be used")
@click.option("--save_dir", type=str, help="directory to output results")
@click.option("--distance", type=int, help="ll param")
@click.option("--steps", type=int, help="ll param")
def ll_random_plane_cli(model_path,
                      dataset,
                      backbone,
                      save_dir,
                      distance,
                      steps):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    ll_random_plane(model_path, dataset, backbone, save_dir, distance, steps)

def accuracy(outputs, labels):
    _, predicted = outputs.max(1)
    total = labels.size(0)
    correct = predicted.eq(labels).sum().item()
    return correct / total

class Acc(loss_landscapes.metrics.Metric):
    """ Computes a specified loss function over specified input-output pairs. """
    def __init__(self, loss_fn, inputs: torch.Tensor, target: torch.Tensor):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.target = target

    def __call__(self, model_wrapper) -> float:
        return self.loss_fn(model_wrapper.forward(self.inputs), self.target)

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
    metric_acc = Acc(accuracy, X, y)
    # compute loss data
    acc_data = loss_landscapes.linear_interpolation(model1, model2, metric_acc, steps, deepcopy_model=True)
    loss_data = loss_landscapes.linear_interpolation(model1, model2, metric, steps, deepcopy_model=True)
    # save loss data
    with open(os.path.join(save_dir, 'linear_interpolation.npy'), 'wb') as f:
        np.save(f, loss_data)
    # plot linear interpolation
    plt.plot([1 / steps * i for i in range(steps)], loss_data, label="loss")
    plt.plot([1 / steps * i for i in range(steps)], acc_data, label="acc")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "linear_interpolation.png"))
    plt.clf()

@click.command()
@click.option("--model_path1", type=str, help="path to model")
@click.option("--dataset", type=str, help="dataset to be used")
@click.option("--backbone", type=str, help="backbone to be used")
@click.option("--save_dir", type=str, help="directory to output results")
@click.option("--steps", type=int, help="ll param")
@click.option("--model_path2", type=str,default=None, help="path to model")
def ll_linear_interpolation_cli(model_path1,
                      dataset,
                      backbone,
                      save_dir,
                      steps,
                      model_path2):
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    ll_linear_interpolation(model_path1, dataset, backbone, save_dir, steps, model_path2)

def gen_plots_fr_np(np_file, save_dir):
    loss_data_fin =np.load(np_file)
    # plot contour
    plt.contour(loss_data_fin, levels=50)
    plt.savefig(os.path.join(save_dir, "contour.png"))
    plt.clf()
    # plot surface
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X = np.array([[j for j in range(loss_data_fin.shape[0])] for i in range(loss_data_fin.shape[0])])
    Y = np.array([[i for _ in range(loss_data_fin.shape[0])] for i in range(loss_data_fin.shape[0])])
    ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    plt.savefig(os.path.join(save_dir, "surface.png"))

def get_best_chromo(outcomes):
    chromo_fitness = []
    for fitness, chromo in zip(outcomes["pop_vals"], outcomes["chromos"]):
        chromo_fitness.append([chromo[1], fitness[1]])
    chromo_fitness.sort(key=lambda x: x[1])
    return chromo_fitness[-1]


def train_model_longer(pretext_model_path,
                       outcome_path,
                       dataset,
                       backbone,
                       ssl_task,
                       ssl_epochs,
                       ssl_batch_size,
                       downstream_epochs,
                       save_dir):
    """
    train model with pretext base for more epochs
    :return:
    """
    # open outcomes file #
    with open(outcome_path, "r") as f:
        outcomes = json.load(f)
    best_chromo, init_fitness = get_best_chromo(outcomes)

    # train using fitness function
    fitness = fitness_function(dataset=dataset,
                               backbone=backbone,
                               ssl_task=ssl_task,
                               ssl_epochs=ssl_epochs,
                               ssl_batch_size=ssl_batch_size,
                               evaluate_downstream_method="finetune",
                               device="cuda",
                               eval_method="best val test",
                               exp_dir=save_dir,
                               evaluate_downstream_kwargs={"num_epochs":downstream_epochs})
    outcomes = []
    model, ssl_losses, train_losses, train_accs, val_losses, val_accs, test_accs, test_losses = fitness(best_chromo,
                                                                                                     return_losses=True,
                                                                                                     pretext_model_ckpt=pretext_model_path,
                                                                                                     eval_test_during_training=True)
    model_path = os.path.join(save_dir, "models")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    torch.save(model.state_dict(), os.path.join(model_path, "model.pt"))
    outcomes.append([
                     ssl_losses[-1],
                     train_losses[-1],
                     val_losses[-1],
                     test_losses[-1],
                     test_accs[-1],
                     train_accs[-1],
                     val_accs[-1]
                     ])
    columns = [
        "final ssl loss",
        "final train loss",
        "final val loss",
        "final test loss",
        "test acc",
        "train acc",
        "val acc"]
    values = [
             ssl_losses,
             train_losses,
             val_losses,
             test_losses,
             test_accs,
             train_accs,
             val_accs
             ]
    outcomes_full = {k:v for k, v in zip(columns,values)}
    with open(os.path.join(save_dir, "outcomes.json"), "w") as f:
        json.dump(outcomes_full, f)
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(train_losses, label="train loss")
    ax[0].plot(val_losses, label="val loss")
    ax[0].plot(test_losses, label="test loss")
    ax[0].legend()
    ax[1].plot(train_accs, label="train acc")
    ax[1].plot(val_accs, label="val acc")
    ax[1].plot(test_accs, label="test acc")
    ax[1].legend()
    plt.savefig(os.path.join(save_dir, f"loss_acc.png"))
    plt.show()
    plt.clf()

    print("final test accuracy: ", test_accs[-1])
    print("final train accuracy: ", train_accs[-1])
    print("final val accuracy: ", val_accs[-1])

    df = pd.DataFrame(outcomes, columns=[
                                         "final ssl loss",
                                         "final train loss",
                                         "final val loss",
                                         "final test loss",
                                         "test acc",
                                         "train acc",
                                         "val acc"])
    df.to_csv(os.path.join(save_dir, "outcomes.csv"))

def train_model_longer_fr_scratch(chromosome,
                                    seed,
                                    dataset,
                                    backbone,
                                    ssl_task,
                                    ssl_epochs,
                                    ssl_batch_size,
                                    downstream_epochs,
                                    save_dir):
    """
    train model with pretext base for more epochs
    :return:
    """

    # train using fitness function
    fitness = fitness_function(dataset=dataset,
                               backbone=backbone,
                               ssl_task=ssl_task,
                               ssl_epochs=ssl_epochs,
                               ssl_batch_size=ssl_batch_size,
                               evaluate_downstream_method="finetune",
                               device="cuda",
                               seed=seed,
                               eval_method="best val test",
                               exp_dir=save_dir,
                               evaluate_downstream_kwargs={"num_epochs":downstream_epochs})
    outcomes = []
    model, ssl_losses, train_losses, train_accs, val_losses, val_accs, test_accs, test_losses = fitness(chromosome,
                                                                                                     return_losses=True,
                                                                                                     eval_test_during_training=True)
    model_path = os.path.join(save_dir, "models")
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    torch.save(model.state_dict(), os.path.join(model_path, "model.pt"))
    outcomes.append([
                     ssl_losses[-1],
                     train_losses[-1],
                     val_losses[-1],
                     test_losses[-1],
                     test_accs[-1],
                     train_accs[-1],
                     val_accs[-1]
                     ])
    columns = [
        "final ssl loss",
        "final train loss",
        "final val loss",
        "final test loss",
        "test acc",
        "train acc",
        "val acc"]
    values = [
             ssl_losses,
             train_losses,
             val_losses,
             test_losses,
             test_accs,
             train_accs,
             val_accs
             ]
    outcomes_full = {k:v for k, v in zip(columns,values)}
    with open(os.path.join(save_dir, "outcomes.json"), "w") as f:
        json.dump(outcomes_full, f)
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(train_losses, label="train loss")
    ax[0].plot(val_losses, label="val loss")
    ax[0].plot(test_losses, label="test loss")
    ax[0].legend()
    ax[1].plot(train_accs, label="train acc")
    ax[1].plot(val_accs, label="val acc")
    ax[1].plot(test_accs, label="test acc")
    ax[1].legend()
    plt.savefig(os.path.join(save_dir, f"loss_acc.png"))
    plt.show()
    plt.clf()

    print("final test accuracy: ", test_accs[-1])
    print("final train accuracy: ", train_accs[-1])
    print("final val accuracy: ", val_accs[-1])

    df = pd.DataFrame(outcomes, columns=[
                                         "final ssl loss",
                                         "final train loss",
                                         "final val loss",
                                         "final test loss",
                                         "test acc",
                                         "train acc",
                                         "val acc"])
    df.to_csv(os.path.join(save_dir, "outcomes.csv"))

def ll_random_plane_v2():
    def tau_2d(alpha, beta, theta_ast):

        a = alpha * theta_ast[:, None, None]
        b = beta * alpha * theta_ast[:, None, None]
        return a + b

    # load model
    backbone = largerCNN_backbone()
    model = finetune_model(backbone.backbone, backbone.in_features, 10)
    model.load_state_dict(torch.load("/home/noah/ESSL/final_exps/optimization/exp8_6/1/models/119_downstream.pt"))

    # get data
    data = Cifar10()
    dataloader = torch.utils.data.DataLoader(data.test_data,
                                             batch_size=len(data.test_data), shuffle=False)
    theta_ast = Params2Vec(model.parameters())

    # backbone = largerCNN_backbone()
    # model_init = finetune_model(backbone.backbone, backbone.in_features, 10)
    # theta = Params2Vec(model_init.parameters())

    loss_fn = torch.nn.CrossEntropyLoss()

    x = torch.linspace(-50, 50, 50)
    y = torch.linspace(-50, 50, 50)
    alpha, beta = torch.meshgrid(x, y)
    alpha, beta = alpha, beta
    space = tau_2d(alpha, beta, theta_ast)

    losses = torch.empty_like(space[0, :, :])
    for a, _ in enumerate(x):
        print(f'a = {a}')
        for b, _ in enumerate(y):
            Vec2Params(space[:, a, b], model.parameters())
            for _, (data, label) in enumerate(dataloader):
                with torch.no_grad():
                    model.eval()
                    losses[a][b] = loss_fn(model(data), label).item()
    losses = losses.numpy()
    print(losses)
    np.save("/home/noah/ESSL/final_exps/optimization/exp8_6/1/ll_medium_approach/50_50_50/plane.npy",
            losses)
if __name__ == "__main__":
    # ll_random_plane(model_path="/home/noah/ESSL/final_exps/optimization/exp8_4/4/models/86_downstream.pt",
    #                 dataset="Cifar10",
    #                 backbone="largerCNN_backbone",
    #                 save_dir="/home/noah/ESSL/final_exps/optimization/exp8_4/4/loss_landscapes",
    #                 distance=10,
    #                 steps=50)
    # gen_plots_fr_np("/home/noah/ESSL/final_exps/optimization/exp8_6/1/loss_landscapes_2/random_plane.npy",
    #                 "/home/noah/ESSL/final_exps/optimization/exp8_6/1/loss_landscapes_2")
    # train_model_longer(pretext_model_path="/home/noah/ESSL/final_exps/optimization/exp8_6/1/models/119_pretext.pt",
    #                    outcome_path="/home/noah/ESSL/final_exps/optimization/exp8_6/1/outcomes.json",
    #                    dataset="Cifar10",
    #                    backbone="largerCNN_backbone",
    #                    ssl_task="SimSiam",
    #                    ssl_epochs=1,
    #                    downstream_epochs=1,
    #                    ssl_batch_size=512,
    #                    save_dir="/home/noah/ESSL/final_exps/optimization/exp8_6/1/train_longer")
    ll_random_plane_v2()