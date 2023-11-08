import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision

learning_rate = 0.001
epochs = 10
batch_size = 200

def load_dataset():
    batch_size = 512 #the image process once

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([ 
                                       torchvision.transforms.ToTensor(), # transfer numpy  to tensor
                                       torchvision.transforms.Normalize( # Data-0.1306 / 0.3081 to normalize data, data distribute around 0
                                           (0.1307,), (0.3081,)) 
                                   ])),
        batch_size=batch_size, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('mnist_data/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def build_model():
    w1, b1= torch.randn(200, 784, requires_grad=True), \
        torch.zeros(200, requires_grad=True)
    w2, b2= torch.randn(200, 200, requires_grad=True), \
        torch.zeros(200, requires_grad=True)
    w3, b3= torch.randn(10, 200, requires_grad=True), \
        torch.zeros(10, requires_grad=True) # bias will broadcasting automatically
    return w1, b1, w2, b2, w3, b3

def forward(x, parameters):
    w1, b1, w2, b2, w3, b3 = parameters
    x = x@w1.t() +b1
    x = F.relu(x)
    x = x@w2.t() +b2
    x = F.relu(x)
    x = x@w3.t() +b3
    x = F.relu(x)

    return x

global parameters


def training(train_loader, criteon):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28 * 28)

        logits = forward(data, parameters)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        loss.backward()
        # print(w1.grad.norm(), w2.grad.norm())
        optimizer.step()

        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def testing(test_loader, criteon):
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data = data.view(-1, 28 * 28)
        logits = forward(data, parameters)
        test_loss += criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':

    train_loader, test_loader = load_dataset()
    w1, b1, w2, b2, w3, b3 = build_model()
    optimizer = optim.SGD([w1, b1, w2, b2, w3, b3], lr=learning_rate)
    criteon = nn.CrossEntropyLoss()
    parameters = [w1, b1, w2, b2, w3, b3]
    for epoch in range(epochs):
        training(train_loader, criteon)
        testing(test_loader, criteon)