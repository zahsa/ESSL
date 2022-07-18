import torch
from torch import nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
from datetime import datetime
import os
import copy


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
    def __call__(self, backbone: torch.nn.Module,
                 device=None,
                 report_all_metrics: bool=False,
                 use_test_acc: bool=True):
        
        if not device:
            device = self.device
        model = finetune_model(backbone.backbone, backbone.in_features, self.dataset.num_classes).to(device)
        trainloader = torch.utils.data.DataLoader(self.dataset.train_data,
                                                  batch_size=self.batch_size,
                                                  shuffle=True,
                                                  num_workers=12
                                                  )
        if self.dataset.val_data:
            valloader = torch.utils.data.DataLoader(self.dataset.val_data,
                                                      batch_size=self.batch_size,
                                                      shuffle=False,
                                                      num_workers = 12
                                                    )
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
        # store best validation for model #
        max_val_acc = -1
        test_model = copy.deepcopy(model)
        if self.verbose:
            epochs = tqdm(range(self.num_epochs))
        else:
            epochs = range(self.num_epochs)
        for epoch in epochs:
            running_loss = 0.0
            correct = 0
            total = 0
            for X, y in trainloader:
                inputs, labels = X.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss+=loss.item()
                loss.backward()
                optimizer.step()
                # record predictions
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            # compute acc
            train_loss = running_loss/len(trainloader)
            train_acc = 100.*correct/total

            # record acc
            train_accs.append(train_acc)
            # record loss
            train_losses.append(train_loss)

            if valloader:
                with torch.no_grad():
                    running_loss = 0.0
                    total = 0
                    correct = 0
                    for X, y in valloader:
                        inputs, labels = X.to(device), y.to(device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        running_loss += loss.item()
                        # record predictions
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()

                # record best acc for testing #
                val_acc = 100.*correct/total
                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                    test_model = copy.deepcopy(model)
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
        if use_test_acc:
            # evaluate #
            # add num workers
            testloader = torch.utils.data.DataLoader(self.dataset.test_data,
                                                     batch_size=self.batch_size,
                                                     shuffle=False,
                                                     num_workers=12)
            test_model.eval()
            total = 0
            correct = 0
            running_loss = 0.0
            # deactivate autograd engine
            with torch.no_grad():
                for X, y in testloader:
                    inputs = X.to(device)
                    labels = y.to(device)
                    outputs = test_model(inputs)
                    loss = criterion(outputs, labels)
                    running_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                test_loss = running_loss / len(testloader)
            test_acc = 100.*correct/total

            if report_all_metrics:
                return train_losses, train_accs, val_losses, val_accs, test_acc, test_loss

            return train_losses, test_acc
        else:
            import pdb;pdb.set_trace()
            if report_all_metrics:
                return train_losses, train_accs, val_losses, val_accs, None, None

            return train_losses, max(val_accs)


