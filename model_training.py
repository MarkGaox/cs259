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
    dataloader = Dataloader(data, batch_size=batch_size)
    return dataloader

def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=False, num_classes=10)
    return model

def get_optimizer(model):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True)
    torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)
    return optimizer

accuracy_loss = nn.CrossEntropyLoss()

class GroupPrune(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'
    
    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        # TODO: lower conv2d weights to matrix, feed to column combining
        # get mask, then reshape mask to conv2d weight mask
        return mask

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
        if epoch == epoches/2:
            for name, module in model.namedModules:
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=0.2)


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    i = 0
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).item()
            i += 1
            if i==1000:
                break
    test_loss /= i
    correct /=i
    print("test loss is {}".format(test_loss))
    print("correct rate is {}".format(correct))


#train(train_dataloader, model, accuracy_loss, \
#        optimizer=optimizer)
hello_world()
#test(test_dataloader, model, accuracy_loss)

#for param in model.parameters():
#    print(type(param), param.size())
#print("named modules:\n")
#for name, module in model.named_modules():
#    print(name, module)
