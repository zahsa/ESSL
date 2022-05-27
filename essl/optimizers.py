from torch import optim


def SGD(model, lr=0.1, momentum=0.9, weight_decay=5e-4):
    return optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay
    )


def Adam(model, lr=3e-4):
    return optim.Adam(
        model.parameters(),
        lr=lr,
    )
