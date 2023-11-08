import torch
import util
from torch import nn
from torch.nn import functional as F
from torch import optim

import torchvision
from matplotlib import pyplot as plt

from util import plot_image, plot_curve, one_hot


# load data
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

# x, y = next(iter(train_loader)) # see data x, and label y
# x = [b, 1, 28, 28] b: batch size

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # xw+b
        self.fc1 = nn.Linear(28*28, 256) # 256 is empirical number
        self.fc2 = nn.Linear(256, 64) # 64 is empirical number
        self.fc3 = nn.Linear(64, 10)
        self.train_loss = []

    def forward(self, x):
        x = F.relu(self.fc1(x)) # h1 = relu(xw+b)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def train(self, train_loader, net):
        
        optimizer = optim.SGD(net.parameters(), lr= 0.01, momentum=0.9) # parameter:[w1, w2, w3, b1, b2, b3]
        for epoch in range(3):
            for batch_idx, (x,y) in enumerate(train_loader):
                x = x.view(x.shape[0], 28*28) # turn [b, 1, 28, 28] => [b, 784]
                out = net(x)
                y_onehot = one_hot(y) # y: [b, 10]
                loss = F.mse_loss(out, y_onehot)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() # w' = w - r*grad
                if batch_idx % 10 == 0:
                    print("epoch {}, batch {}, the loss is {}".format(epoch, batch_idx, loss.item()))
                    self.train_loss.append(loss.item())

    def validate(self, test_loader, net):
        total_correct = 0
        for x, y in test_loader:
            x = x.view(x.shape[0], 28*28)
            out = net(x) # [b, 10]
            pre = out.argmax(dim=1) # [b]
            correct = pre.eq(y).sum().float().item()
            total_correct += correct
        total_num = len(test_loader.dataset)
        acc = total_correct / total_num

        return acc


if __name__ == "__main__":
   train, valid =  load_dataset()
   net = Net()
   net.train(train, net)
   print(net.validate(valid, net)) # accuracy 90.1%
