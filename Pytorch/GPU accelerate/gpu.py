import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device('cuda:0')
net = MLP().to(device)
optimize = optim.SGC(net.parameters(), lr=learning_rate)
criteon = nn.CrossEntropyLoss().to(device)

for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 28*28)
        data, target = data.to(device), target.cuda() # to() new recommend method, cuda() old version

