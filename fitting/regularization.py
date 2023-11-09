import torch
from torch import optim

# L2 loss
optimizer = optim.SGD(net.parameters(), lr=leanring_rate, weight_decay=0.01)

# L1 loss
regularization_loss = 0
for param in model.parameters():
    regularization_loss += torch.sum(torch.abs(param))

classify_loss = criteon(logits, target)
loss = classify_loss + 0.01 * regularization_loss

