import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torch.nn.utils.prune as prune
import torch.nn.functional as F

from column_combine import *
batch_size = 32
epoches = 300

def load_data(task_type):
    if task_type == "train":
        train = True
    elif task_type == "test":
        train = False
    else:
        raise ValueError
    data = datasets.CIFAR10(root='./data', train=train, download=True, transform=ToTensor(),)
    dataloader = DataLoader(data, batch_size=batch_size)
    return dataloader

def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=False, num_classes=10)
    return model

def get_optimizer(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)
    return optimizer

accuracy_loss = nn.CrossEntropyLoss()

class GroupPruneMethod(prune.BasePruningMethod):
    PRUNING_TYPE = 'structured'
    dim = -1

    def compute_mask(self, t, default_mask):
        matrix = t.reshape([t.shape[0], -1]).detach().numpy()
        groups = columnCombine(matrix)
        mask = structuredPruneMask(matrix, groups)
        mask = torch.tensor(mask.reshape(t.shape))
        return mask


def group_prune_structured(module, name):
    GroupPruneMethod.apply(module, name)
    return module

"""
Test how to correctly use custom prune method

class TrivialPruneMethod(prune.BasePruningMethod):
    PRUNING_TYPE = 'structured'
    dim = -1
    def compute_mask(self, t, default_mask):
        matrix = t.reshape([t.shape[0], -1]).detach().numpy()
        mask = np.zeros_like(matrix)
        mask[:, t.shape[1]**2:t.shape[1]**2 + 3] = 1
        #print("matrix shape mask:\n", mask)
        mask = torch.tensor(mask.reshape(t.shape))
        #print("tensor shape mask:\n", mask)
        return mask

def trivial_prune_structured(module, name):
    TrivialPruneMethod.apply(module, name)
    return module
"""

def train(dataloader, model, loss_fn, optimizer, regularization=True, lambda1 = 1e-7):
    model.train()
    for epoch in range(epoches):
        print("epoch {} begins:".format(epoch))
        for batch, (X,y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_fn(pred, y)
            if regularization:
                l1_params = torch.cat([x.view(-1) for x in model.parameters()])
                loss += lambda1 * torch.norm(l1_params, 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch % 10 ==0):
                print("pass {}, loss is {}".format(batch, loss.item()))
                break
        if epoch == epoches/2 or True:
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=0.2)
                    print("named buffer numbers\n", len(list(module.named_buffers())))
                    group_prune_structured(module, name='weight')
                    # trivial_prune_structured(module, name='weight')
                    print("named buffer numbers\n", len(list(module.named_buffers())))
        break

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).item()
    test_loss /= size
    correct /= size
    print("test loss is {}".format(test_loss))
    print("correct rate is {}".format(correct))


train_dataloader = load_data("train")
model = load_model()
optimizer = get_optimizer(model)

train(train_dataloader, model, accuracy_loss, \
        optimizer=optimizer)
#test(test_dataloader, model, accuracy_loss)
