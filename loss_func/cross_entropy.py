import torch
from torch.nn import functional as F

a = torch.tensor([0.01, 0.01, 0.05, 0.38])
print("cross entropy: ", -(a*torch.log2(a)).sum())
b = torch.tensor([0.01, 0.01, 0.01, 0.998])
print("cross entropy sum: ", -(a*torch.log2(b)).sum())
print("cross entropy sum: ", -(a*torch.log2(b).t()).sum()) # will auto transpose

x = torch.randn(1, 784)
w = torch.randn(10, 784)

logits = x @ w.t()
pred = F.softmax(logits, dim=1)

pred_log = torch.log(pred)
print("pred_log: ", pred_log)

print(F.cross_entropy(logits, torch.tensor([3]))) # the loss function has integrate a softmax
print(F.nll_loss(pred_log, torch.tensor([3]))) # or use none negative loss

