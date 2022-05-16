import torchvision
import torch
from torch import nn
from sklearn.metrics import accuracy_score

from essl import optimizers
from essl import losses
from essl import datasets

class finetune_model(nn.Module):
    def __init__(self, backbone, in_features, num_outputs):
        super().__init__()
        self.backbone = backbone
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=num_outputs, bias=True),
        )
    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.classifier(x)
        return x

class finetune:
    def __init__(self,
                    dataset: datasets.Data,
                    opt: str = "SGD",
                    num_epochs: int = 10,
                    loss: str = "CrossEntropyLoss",
                    batch_size: int = 32,
                    device: str = "cuda"):
        self.dataset = dataset
        self.opt = optimizers.__dict__[opt]
        self.num_epochs = num_epochs
        self.loss = losses.__dict__[loss](device)
        self.batch_size = batch_size
        self.device = device

    def __call__(self, backbone):
        model = finetune_model(backbone.backbone, backbone.in_features, self.dataset.num_classes).to(self.device)
        trainloader = torch.utils.data.DataLoader(self.dataset.train_data,
                                                  batch_size=self.batch_size)
        criterion = self.loss
        optimizer = self.opt(model)
        # train #
        losses = []
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            for X, y in trainloader:
                inputs, labels = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss+=loss.item()
                loss.backward()
                optimizer.step()
            losses.append(running_loss/len(trainloader))
        # evaluate #
        testloader = torch.utils.data.DataLoader(self.dataset.test_data,
                                                 batch_size=self.batch_size)
        model.eval()
        y_true = torch.tensor([], dtype=torch.long).to(self.device)
        pred_probs = torch.tensor([]).to(self.device)
        # deactivate autograd engine
        with torch.no_grad():
            running_loss = 0.0
            for X, y in testloader:
                inputs = X.to(self.device)
                labels = y.to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                y_true = torch.cat((y_true, labels), 0)
                pred_probs = torch.cat((pred_probs, outputs), 0)

        y_true = y_true.cpu().numpy()
        _, y_pred = torch.max(pred_probs, 1)
        y_pred = y_pred.cpu().numpy()
        # return acc
        return losses, accuracy_score(y_true, y_pred)


