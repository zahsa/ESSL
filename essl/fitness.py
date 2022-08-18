import os
import numpy as np
import torchvision
import torch
from lightly.data import LightlyDataset
import glob

from essl import ops
from essl import chromosome
from essl import pretext_selection
from essl import backbones
from essl import datasets
from essl import evaluate_downstream
import time

def dummy_eval(chromosome):
    """
    dummy evaluation technique, order the augmentations sequentially
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
    def __call__(self,
                 transform,
                 device=None
                 ):
        if not device:
            device = self.device
        torch.manual_seed(self.seed)
        backbone = self.backbone()
        torch.manual_seed(self.seed)
        model = self.model(backbone)
        loss = model.fit(
                  self.dataset,
                  self.batch_size,
                  self.num_epochs,
                  transform,
                  device
                         )
        return model, loss


class fitness_function:
    """
    proposed approach:
    wrap above workflow in a class to store global aspects of the evaluation such as
    dataset and hparams
    """
    def __init__(self,
                 dataset: str,
                 exp_dir: str,
                 backbone: str,
                 ssl_task: str,
                 ssl_epochs: int,
                 ssl_batch_size: int,
                 evaluate_downstream_method: str,
                 evaluate_downstream_kwargs: dict = {},
                 device: str = "cuda",
                 seed: int=10,
                 eval_method: str = "final test"):

        # set seeds #
        self.seed = seed
        # torch.cuda.manual_seed_all(self.seed)
        # torch.cuda.manual_seed(self.seed)
        # torch.manual_seed(self.seed)

        self.dataset = datasets.__dict__[dataset](seed=seed)
        self.exp_dir = exp_dir
        self.model_dir = os.path.join(self.exp_dir, "models")
        if not os.path.isdir(self.model_dir):
            os.mkdir(self.model_dir)
        self.backbone = backbone
        self.ssl_epochs = ssl_epochs
        self.ssl_batch_size = ssl_batch_size
        self.evaluate_downstream_method = evaluate_downstream_method
        self.evaluate_downstream_kwargs = evaluate_downstream_kwargs
        self.ssl_task = ssl_task
        self.downstream_losses = {}
        self.device = device
        self.eval_method = eval_method
        self.ssl_task = pretext_task(method=self.ssl_task,
                                dataset=self.dataset,
                                backbone=self.backbone,
                                num_epochs=self.ssl_epochs,
                                batch_size=self.ssl_batch_size,
                                device=self.device,
                                seed=self.seed
                                )
        self.evaluate_downstream = evaluate_downstream.__dict__[self.evaluate_downstream_method](dataset=self.dataset,
                                                                                                 seed=self.seed,
                                                                                                 device=self.device,
                                                                                                 **self.evaluate_downstream_kwargs)

        self.best_chromo_info = {
            "id":0,
            "fitness":-1
        }

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

    def evaluate_default_aug(self, device=None,
                                   return_losses=False,
                                   verbose=True):
        if not device:
            device = self.device
        t1 = time.time()
        # pass none to transform to get default aug
        representation, ssl_losses = self.ssl_task(transform=None,
                                                   device=device
                                                   )
        # return all metrics by default
        train_losses, train_accs, val_losses, val_accs, test_acc, test_loss = self.evaluate_downstream(representation,
                                                                                                       # device=device,
                                                                                                       report_all_metrics=True,
                                                                                                       eval_method=self.eval_method)
        if verbose:
            print("time to eval: ", time.time() - t1)
        if return_losses:
            return ssl_losses, train_losses, train_accs, val_losses, val_accs, test_acc, test_loss
        else:
            # store the losses with id of chromosome, try statement to allow for use outside of GA
            try:
                self.downstream_losses[chromosome.id] = train_losses
            except:
                pass
            # default no return losses,
            if self.eval_method in ["best val test", "final test"]:
                return test_acc,

            else:
                return max(val_accs),
    def clear_models(self):
        opt_model = os.path.join(self.model_dir, str(self.best_chromo_info["id"]) + ".pt")
        for f in glob.glob(os.path.join(self.model_dir, "*.pt")):
            if f != opt_model:
                os.remove(f)
    def save_best_model(self, model_path):
        print(f"saving model to {model_path}")
        torch.save(self.best_chromo_info["model"], model_path)

    def __call__(self, chromosome,
                 device=None,
                 return_losses=False,
                 verbose=True):
        if not device:
            device = self.device
        t1 = time.time()
        transform = self.gen_augmentation_torch(chromosome)

        representation, ssl_losses = self.ssl_task(transform,
                                                   device=device
                                                   )
        # set save path for model
        # self.evaluate_downstream.save_best_model_path = os.path.join(self.model_dir, str(chromosome.id))+".pt"
        model, train_losses, train_accs, val_losses, val_accs, test_acc, test_loss = self.evaluate_downstream(representation,
                                                                                                       # device=device,
                                                                                                       report_all_metrics=True,
                                                                                                       eval_method=self.eval_method)
        if verbose:
            print("time to eval: ", time.time() - t1)
        if return_losses:
            return ssl_losses, train_losses, train_accs, val_losses, val_accs, test_acc, test_loss
        else:
            # store the losses with id of chromosome, try statement to allow for use outside of GA
            try:
                self.downstream_losses[chromosome.id] = train_losses
            except:
                pass
            # default no return losses,
            if self.eval_method in ["best val test", "final test"]:
                if test_acc > self.best_chromo_info["fitness"]:
                    self.best_chromo_info["id"] = chromosome.id
                    self.best_chromo_info["fitness"] = test_acc
                    self.best_chromo_info["model"] = model.state_dict()
                    self.clear_models()
                    self.save_best_model(model_path=os.path.join(self.model_dir, str(chromosome.id))+".pt")
                return test_acc,

            else:
                if max(val_accs) > self.best_chromo_info["fitness"]:
                    self.best_chromo_info["id"] = chromosome.id
                    self.best_chromo_info["fitness"] = max(val_accs)
                    self.clear_models()
                    self.save_best_model(model_path=os.path.join(self.model_dir, str(chromosome.id))+".pt")
                return max(val_accs),

class fitness_function_mo:
    """
    proposed approach:
    wrap above workflow in a class to store global aspects of the evaluation such as
    dataset and hparams
    """

    # D1: remove ssl task as option from fitness function
    def __init__(self,
                 dataset: str,
                 backbone: str,
                 ssl_epochs: int,
                 ssl_batch_size: int,
                 evaluate_downstream_method: str,
                 evaluate_downstream_kwargs: dict = { },
                 device: str = "cuda",
                 seed: int = 10,
                 eval_method="final test"):

        # set seeds #
        self.seed = seed
        # torch.cuda.manual_seed_all(self.seed)
        # torch.cuda.manual_seed(self.seed)
        # torch.manual_seed(self.seed)

        self.dataset = datasets.__dict__[dataset](seed=seed)
        self.backbone = backbone
        self.ssl_epochs = ssl_epochs
        self.ssl_batch_size = ssl_batch_size
        self.evaluate_downstream_method = evaluate_downstream_method
        self.evaluate_downstream_kwargs = evaluate_downstream_kwargs
        # self.ssl_task = ssl_task
        self.downstream_losses = { }
        self.device = device
        self.eval_method = eval_method
        self.evaluate_downstream = evaluate_downstream.__dict__[self.evaluate_downstream_method](
            dataset=self.dataset,
            seed=self.seed,
            device=self.device,
            **self.evaluate_downstream_kwargs)

    @staticmethod
    def gen_augmentation_torch(chromosome: list) -> torchvision.transforms.Compose:
        # gen augmentation
        transform = torchvision.transforms.Compose([
                                                       torchvision.transforms.Lambda(ops.__dict__[op](intensity=i))
                                                       for op, i in chromosome
                                                   ] + [torchvision.transforms.ToTensor()])
        return transform

    def clear_downstream_losses(self):
        self.downstream_losses = { }

    def __call__(self, chromosome,
                 device=None,
                 return_losses=False):
        if not device:
            device = self.device
        ssl_task = pretext_task(method=chromosome[0],
                                dataset=self.dataset,
                                backbone=self.backbone,
                                num_epochs=self.ssl_epochs,
                                batch_size=self.ssl_batch_size,
                                device=self.device,
                                seed=self.seed
                                )
        t1 = time.time()
        transform = self.gen_augmentation_torch(chromosome[1:])
        representation, ssl_losses = ssl_task(transform,
                                              device=device
                                              )
        train_losses, train_accs, val_losses, val_accs, test_acc, test_loss = self.evaluate_downstream(
                                                                        representation,
                                                                        eval_method=self.eval_method,
                                                                        report_all_metrics=True)

        print("time to eval: ", time.time() - t1)
        if return_losses:
            return ssl_losses, train_losses, train_accs, val_losses, val_accs, test_acc, test_loss
        else:
            # store the losses with id of chromosome, try statement to allow for use outside of GA
            try:
                self.downstream_losses[chromosome.id] = train_losses
            except:
                pass
            # default no return losses,
            if self.eval_method in ["best val test", "final test"]:
                return test_acc,

            else:
                return max(val_accs),

if __name__ == "__main__":

    """
    testing to see that the model weights are fixed, note that to do this the pretext task function
    
    returns both the backbone and model immediately after it is created (line 52), subsequently the fitness function
    reutrns both of these models (line 167).
    """

    c = chromosome.chromosome_generator()
    cc = c()
    fitness = fitness_function(dataset="Cifar10",
                                 backbone="largerCNN_backbone",
                                 ssl_task="SimCLR",
                                 ssl_epochs=1,
                                 ssl_batch_size=64,
                                 evaluate_downstream_method="finetune",
                                 evaluate_downstream_kwargs={"num_epochs":4,
                                                             "save_best_model_path": "/home/noah/ESSL/exps/testing/no_flip_ops/model.pt"},
                                 device="cuda",
                                 eval_method="best val test")
    bb1, model1 = fitness(cc, return_losses=True)
    bb2, model2 = fitness(cc, return_losses=True)

    def compare_models(model_1, model_2):
        models_differ = 0
        for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
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

    compare_models(bb1, bb2)
    compare_models(model1, model2)
    """
    Models match perfectly! :)
    Models match perfectly! :)
    """


