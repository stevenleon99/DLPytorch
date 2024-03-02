import torch
import numpy as np

################################tensor#################################
a = torch.randn(2,3) # random normalize distribu
b = torch.rand(2,3) # random uniform distribu
print(a.type())

scale = torch.tensor(1.0)
print(scale.type())
print("scalar's dimension: ", scale.dim())

tensor = torch.tensor([2])
print("tensor's dimension: ", tensor.dim())
print(tensor.item())

data = np.ones(2)
data = torch.from_numpy(data)

a = torch.ones(2,3)
print(a, a.shape, a.size(0), a.size(1))

a = torch.rand(1,2,3)
print(a)
print(a.shape) # three dims: RNN NLP processing, (word, sentence, features)

a = torch.rand(2,3,28,28) # four dims: num of photo, channel, h, w
print(a)


################################create_tensor#################################
uninit = torch.Tensor(3,2) # create unitialize tensor
print(a)

em = torch.empty(2,3)
print(a)
# torch.set_default_tensor_type(torch.DoubleTensor) #the unindicated create float will become double

a_like = torch.rand_like(a)
print(a.shape)

a_range = torch.randint(1, 10, (3,3)) # min 1 max 1 shape 3 3
print(a_range)

a_fill_value = torch.full([2,2], 0)
scalar_fill_value = torch.full([], 0)

sequnece = torch.arange(0, 10) # include left exclude 10

linspace = torch.linspace(0, 10, steps=2) # include left and right

log_lin = torch.logspace(0, -1, steps=10) # base is 10 by default

one = torch.ones(3,3)
zero = torch.zeros(3,3)
eye = torch.eye(3)

randon_index = torch.randperm(10)
print(randon_index)

################################slice#################################
print(a[:,:,:-4,:].shape)


################################dimension change#################################
a=torch.rand(4,1,28,28)
print(a.view(4,28,28).size()) # torch 3.1 or newer, view is same with reshape
print(a.unsqueeze(2).shape) # insert a dimension at index 2
print(a.squeeze(0).shape) # squeeze a dimension at index 0, if the shape not 1 then unchange
print(a.unsqueeze(2).expand(4,1,10,28,28).shape)

ashape = a.transpose(1,3).shape
b = a.transpose(1,3)
print(b.contiguous().view(4*28, 28, 1)) # after transpose, lose the continous
b_permu = a.permute(0, 2, 3, 1) # permute can change one than 1 pair
print(b_permu.shape)


################################broadcasting#################################
A = torch.tensor([1.0,2.0,3.0])
print(A, A.shape)
A_bc = A.broadcast_to(3,3)
print(A_bc)
# [class, students, scores] give every student +5
score = torch.randint(0,50, (4, 32, 8))
print(score, score.shape)
add = torch.tensor([5.0])
add_bc = add.broadcast_to(4,32,8)
score_add = score + add
print(score_add)


################################cat stack split trunk#################################
a =torch.rand(4,32,8)
b = torch.rand(5,32,8)
print(torch.cat([a,b], dim = 0).shape)

a = torch.rand(32, 8)
b = torch.rand(32, 8)
print(torch.stack((a,b), dim=0).shape) # new dimension to combine two tensors
ab = torch.stack((a,b), dim=0)
a, b = ab.split(1, dim=0) # 1 is the number assigned to each block, can be [ ] if different for each block
print(a.shape)
a, b = ab.chunk(2, dim=0) # 2 is the number of block
