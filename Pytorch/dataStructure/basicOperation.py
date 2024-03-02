import torch
import numpy as np

A = torch.rand(3,4)
B = torch.full((4,3),1.0)
print(B)
print(torch.matmul(A,B))
print(torch.all(torch.eq(torch.mm(A,B),torch.matmul(A,B))))
# the torch.mm can be used only for two dimensions and not broadcastible
# the torch.matmul can be used for more than two and will broadcast
# [4,3,24, 65] torch.matmul with [4,3, 65,24], the axis 2 will broadcast to 3


A = torch.rand(3,4,2)
B = torch.rand(1,2,4)
print(torch.matmul(A,B).shape)


print(torch.all(torch.eq(A.pow(2), A**2)))
fl = torch.randn(3,4)*15
print(fl.trunc().clamp(0))

################################stat#################################

a = torch.full([8], 1.0)
a = a.view(2,4)
print(a)
print(a.norm(2, dim=1)) # para 2 is the type of norm; param dim=1 is the calculation dimension

A = torch.arange(8).view(2,4).float()
print(A.prod()) # prod is the cumulative multiplication
print(A.argmin(dim= 1)) # return the index
print(A.max(dim=1)) # values=tensor([3., 7.]), indices=tensor([3, 3]))
print(A.max(dim=1, keepdim=True)) # values=tensor([[3.],[7.]]),indices=tensor([[3],[3]]))
print(A.topk(3, dim=1))
print(A.topk(3, dim=1, largest=False)) # the 3 smallest number and the index
print(A.kthvalue(2, dim=1)) # the 2nd smallest number and the index


print(torch.gt(a,0))


################################where gather#################################
con = torch.rand(2,2)
a = torch.full((2,2), 0.0)
b = torch.full((2,2), 1.0)
print(con)
print(torch.where(con>0.5, a, b))

prob = torch.randn(4,10)
idx = prob.topk(dim = 1, k=3)[1]
print(idx)
label = torch.arange(10)+100
print(label)
res = torch.gather(label.expand(4, 10), dim=1, index=idx.long()) # like a table looking operation: look for the number in table of "label"
print(res)

