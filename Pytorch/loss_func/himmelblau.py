import matplotlib.pyplot as plt
import torch 
from torch.nn import functional as F
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def him(x):
     return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 -7)**2

x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
X, Y = np.meshgrid(x,y)
print(X.shape, Y.shape)
input = [X, Y] # list all the permutation of x and y
Z = him([X, Y])
print(Z.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

x = torch.tensor([0., 0.], requires_grad=True)
optim = torch.optim.Adam([x], lr=1e-3)
for step in range(20000):
     pred = him(x)

     optim.zero_grad()
     pred.backward()
     optim.step()

     if step % 200 == 0:
          print('step{}: x = {}, f(x) = {}'.format(step, x.tolist(), pred))