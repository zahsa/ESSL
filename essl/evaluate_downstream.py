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
                    save_best_model_path: str = None,
                    use_scheduler: bool = False,
                    seed: int = 10,
                    ):
        self.dataset = dataset
        self.opt = optimizers.__dict__[opt]
        self.num_epochs = num_epochs
        self.loss = losses.__dict__[loss](device)
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose
        self.seed = seed
        self.save_best_model_path = save_best_model_path

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
                 eval_method: str="best val test"):
        torch.manual_seed(self.seed)
        model = finetune_model(backbone.backbone, backbone.in_features, self.dataset.num_classes).to(self.device)
        trainloader = torch.utils.data.DataLoader(self.dataset.train_data,
                                                  batch_size=self.batch_size, shuffle=True)
        if self.dataset.val_data:
            valloader = torch.utils.data.DataLoader(self.dataset.val_data,
                                                    batch_size=self.batch_size, shuffle=False)
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
        max_val_acc = -1
        max_val_model = copy.deepcopy(model.state_dict())
        if self.verbose:
            epochs = tqdm(range(self.num_epochs))
        else:
            epochs = range(self.num_epochs)
        for epoch in epochs:
            running_loss = 0.0
            correct = 0
            total = 0
            for X, y in trainloader:
                inputs, labels = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
                # record predictions
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
            # compute acc
            train_loss = running_loss / len(trainloader)
            train_acc = 100. * correct / total

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
                        inputs, labels = X.to(self.device), y.to(self.device)
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        running_loss += loss.item()
                        # record predictions
                        _, predicted = outputs.max(1)
                        total += labels.size(0)
                        correct += predicted.eq(labels).sum().item()
                # compute acc
                val_acc = 100. * correct / total
                if val_acc > max_val_acc:
                    max_val_acc = val_acc
                    # deepcopy model weights so we do not refer to the same statedict
                    max_val_model = copy.deepcopy(model.state_dict())
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

        if eval_method == "best val acc":
            model.load_state_dict(max_val_model)
        model.eval()
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
        test_acc = 100. * correct / total
        # save model #
        if self.save_best_model_path:
            print(f"saving model to {self.save_best_model_path}")
            torch.save(model.state_dict(), self.save_best_model_path)
        if report_all_metrics:
            return model, train_losses, train_accs, val_losses, val_accs, test_acc, test_loss

        return train_losses, test_acc


if __name__ == "__main__":
    from essl import backbones
    from datasets import Cifar10
    import torch
    """
    confirm that when we use the ft twice on two generated models using
    manual seed, the exact same networks are produced.
    
    Note when this run the model was returned on line 73 of fitness function right 
    after it was instantiated
    
    """
    ft = finetune(Cifar10(), num_epochs = 1)
    torch.manual_seed(10)
    backbone1 = backbones.largerCNN_backbone()
    model1 = ft(backbone1, eval_method = "best val acc")
    # torch.manual_seed(10)
    # backbone2 = backbones.largerCNN_backbone()
    # model2 = ft(backbone2)
    # def compare_models(model_1, model_2):
    #     models_differ = 0
    #     for key_item_1, key_item_2 in zip(model_1.state_dict().items(), model_2.state_dict().items()):
    #         if torch.equal(key_item_1[1], key_item_2[1]):
    #             pass
    #         else:
    #             models_differ += 1
    #             if (key_item_1[0] == key_item_2[0]):
    #                 print('Mismtach found at', key_item_1[0])
    #             else:
    #                 raise Exception
    #     if models_differ == 0:
    #         print('Models match perfectly! :)')
    #
    #
    # compare_models(model1, model2)