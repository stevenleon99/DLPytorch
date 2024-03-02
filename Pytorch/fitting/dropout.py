import torch.nn as nn
import torch

net_dropped = torch.nn.Sequential(
    nn.Linear(784, 200),
    nn.Dropout(0.5), # drop 50% of the neurons
    torch.nn.ReLU(),
    torch.nn.Linear(200,200),
    nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(200, 100)
)


###in the test
net_dropped.eval() # need to disable dropout when testing
test_loss = 0
correct = 0
for data, target in test_loader:
    pass

