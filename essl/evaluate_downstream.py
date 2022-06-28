import pdb

import torchvision
import torch
from torch import nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import torch.nn.functional as F

from essl import optimizers
from essl import losses
from essl import datasets

class finetune_model(nn.Module):
    def __init__(self, backbone, in_features, num_outputs, linear=False):
        super().__init__()
        self.backbone = backbone
        if linear:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features=in_features, out_features=num_outputs, bias=True),
            )
        else:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features=in_features, out_features=1024, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(in_features=1024, out_features=512, bias=True),
                torch.nn.ReLU(),
                nn.Linear(512, num_outputs)
            )
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.classifier(x)
        return x

class finetune:
    def __init__(self,
                    dataset: datasets.Data,
                    opt: str = "Adam",
                    num_epochs: int = 10,
                    loss: str = "CrossEntropyLoss",
                    batch_size: int = 32,
                    device: str = "cuda",
                    verbose: bool = False,
                    tensorboard_dir: str = None,
                    use_scheduler: bool = False,
                    seed: int = 10):
        self.dataset = dataset
        self.opt = optimizers.__dict__[opt]
        self.num_epochs = num_epochs
        self.loss = losses.__dict__[loss](device)
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.seed = seed
        # set seeds #
        torch.cuda.manual_seed_all(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.manual_seed(self.seed)

        if tensorboard_dir:
            if not os.path.isdir(tensorboard_dir):
                os.mkdir(tensorboard_dir)
            self.writer = SummaryWriter(os.path.join(tensorboard_dir, datetime.now().strftime("%Y%m%d_%H:%M")))
        else:
            self.writer = None
        self.use_scheduler = use_scheduler

    def __call__(self, backbone: torch.nn.Module, report_all_metrics: bool=False):
        model = finetune_model(backbone.backbone, backbone.in_features, self.dataset.num_classes).to(self.device)
        trainloader = torch.utils.data.DataLoader(self.dataset.train_data,
                                                  batch_size=self.batch_size, shuffle=True, num_workers=0)
        if self.dataset.val_data:
            valloader = torch.utils.data.DataLoader(self.dataset.val_data,
                                                      batch_size=self.batch_size, shuffle=False, num_workers=0)
        else:
            valloader = None
        criterion = self.loss
        optimizer = self.opt(model)
        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
        else:
            scheduler = None
        # train #
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []
        if self.verbose:
            epochs = tqdm(range(self.num_epochs))
        else:
            epochs = range(self.num_epochs)
        for epoch in epochs:
            running_loss = 0.0
            # train_y_true = torch.tensor([], dtype=torch.long).to(self.device)
            # train_pred_probs = torch.tensor([]).to(self.device)
            correct = 0
            total = 0
            for X, y in trainloader:
                inputs, labels = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss+=loss.item()
                loss.backward()
                optimizer.step()
                # record predictions
                # train_y_true = torch.cat((train_y_true, labels), 0)
                # train_pred_probs = torch.cat((train_pred_probs, outputs), 0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            # compute acc
            #train_y_true = train_y_true.cpu().numpy()
            #_, train_y_pred = torch.max(train_pred_probs, 1)
            #train_y_pred = train_y_pred.cpu().numpy()
            # train_acc = accuracy_score(train_y_true, train_y_pred)
            train_loss = running_loss/len(trainloader)
            train_acc = 100.*correct/total

            # record acc
            train_accs.append(train_acc)
            # record loss
            train_losses.append(train_loss)

            if valloader:
                with torch.no_grad():
                    # val_y_true = torch.tensor([], dtype=torch.long).to(self.device)
                    # val_pred_probs = torch.tensor([]).to(self.device)
                    running_loss = 0.0
                    total = 0
                    correct = 0
                    for X, y in valloader:
                        inputs, labels = X.to(self.device), y.to(self.device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        running_loss += loss.item()
                        # record predictions
                        # val_y_true = torch.cat((val_y_true, labels), 0)
                        # val_pred_probs = torch.cat((val_pred_probs, outputs), 0)
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                # compute acc
                # val_y_true = val_y_true.cpu().numpy()
                # _, val_y_pred = torch.max(val_pred_probs, 1)
                # val_y_pred = val_y_pred.cpu().numpy()
                # record acc
                # val_acc = accuracy_score(val_y_true, val_y_pred)
                val_acc = 100.*correct/total
                val_accs.append(val_acc)
                # record loss
                val_loss = running_loss / len(valloader)
                val_losses.append(val_loss)
            # tensorboard
            if self.writer:
                self.writer.add_scalar('train/loss', train_loss, epoch)
                self.writer.add_scalar('train/acc', train_acc, epoch)
                if valloader:
                    self.writer.add_scalar('val/loss', val_loss, epoch)
                    self.writer.add_scalar('val/acc', val_acc, epoch)
            if scheduler:
                scheduler.step()

        # evaluate #
        testloader = torch.utils.data.DataLoader(self.dataset.test_data,
                                                 batch_size=self.batch_size,
                                                 shuffle=False)
        model.eval()
        # y_true = torch.tensor([], dtype=torch.long).to(self.device)
        # pred_probs = torch.tensor([]).to(self.device)
        total = 0
        correct = 0
        running_loss = 0.0
        # deactivate autograd engine
        with torch.no_grad():
            for X, y in testloader:
                inputs = X.to(self.device)
                labels = y.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                # y_true = torch.cat((y_true, labels), 0)
                # pred_probs = torch.cat((pred_probs, outputs), 0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            test_loss = running_loss / len(testloader)


        # y_true = y_true.cpu().numpy()
        # _, y_pred = torch.max(pred_probs, 1)
        # y_pred = y_pred.cpu().numpy()
        test_acc = 100.*correct/total

        if report_all_metrics:
            return train_losses, train_accs, val_losses, val_accs, test_acc, test_loss

        return train_losses, test_acc


