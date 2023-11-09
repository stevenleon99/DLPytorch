import torch
from torch import optim

# L2 loss
optimizer = optim.SGD(net.parameters(), 
                      lr=leanring_rate, 
                      weight_decay=0.01,
                      momentum=args.momentum)


