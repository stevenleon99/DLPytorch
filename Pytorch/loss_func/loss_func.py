import torch
from torch.nn import functional as F

x = torch.ones(1, requires_grad=True)
w = torch.full([1], 2.0, requires_grad=True)
mse = F.mse_loss(torch.ones(1), x.mul(w))
# print(torch.autograd.grad(mse, [w]))
mse.backward()
print(x.grad, w.grad)

x = torch.rand(1,10, requires_grad=True)
w = torch.rand(1,10, requires_grad=True)
out = torch.sigmoid(torch.matmul(x, w.t()))
print(out)
loss = F.mse_loss(torch.ones(1,1), out)
print(loss)
loss.backward()
print(w.grad)
