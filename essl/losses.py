from torch import nn

def CrossEntropyLoss(device):
    return nn.CrossEntropyLoss().to(device)