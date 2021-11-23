import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

batch_size = 32

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=ToTensor(),)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=ToTensor(),)

train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in train_dataloader:
    print("Shape of X is: ", X.shape)
    print("Shape of y: ", y.shape, y.dtype)
    break

model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=False, num_classes=10)



lambda1 = 0.5

optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

accuracy_loss = nn.CrossEntropyLoss()

all_linear1_params = torch.cat([x.view(-1) for x in model.parameters()])
l1_regularization = lambda1 * torch.norm(all_linear1_params, 1)


def train(dataloader, model, loss_fn, optimizer, regularization):
    model.train()
    i = 0
    for batch, (X,y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
        if (i % 10 ==0):
            print("pass {}, loss is {}".format(i, loss.item()))
        if i == 1000:
            break

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


train(train_dataloader, model, accuracy_loss, \
        optimizer=optimizer, regularization = l1_regularization)
test(test_dataloader, model, accuracy_loss)

#for param in model.parameters():
#    print(type(param), param.size())
#for i, module in enumerate(model.modules()):
#    if (isinstance(module, nn.Conv2d)):
#        print(module)
        #print(module.bias)
        #print(module.weight)
