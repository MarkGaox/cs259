import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torch.nn.utils.prune as prune
import prune_method as customized_prune

batch_size = 128
epoches = 100

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

def save_model(model, epoch):
    print('save model for epoch {}'.format(epoch))
    torch.save(model, "./models/epoch_{}".format(epoch))


def train(dataloader, model, loss_fn, optimizer, regularization=True, lambda1 = 1e-7):
    model.train()
    for epoch in range(epoches):
        print("epoch {} begins:".format(epoch))
        save_model(model, 0)
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
        if epoch == epoches/2 :
            for name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module, name='weight', amount=0.3)
                    customized_prune.group_prune_structured(module, name='weight')
                    # trivial_prune_structured(module, name='weight')
        if (epoch + 1) % 10 == 0:
            save_model(model, epoch)

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
