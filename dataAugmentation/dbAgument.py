import torch
import torchvision
from torch.nn import Transformer
from torchvision import transforms

batch_size = 50

train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('mnist_data', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.RandomHorizontalFlip(),
                                       torchvision.transforms.RandomVerticalFlip(),
                                       torchvision.transforms.RandomRotation(15), # rotate from -15<0<15
                                       torchvision.transforms.RandomRotation([90, 180, 270]), # rotate from three angles
                                       torchvision.transforms.ToTensor(), # transfer numpy  to tensor
                                       torchvision.transforms.Normalize( # Data-0.1306 / 0.3081 to normalize data, data distribute around 0
                                           (0.1307,), (0.3081,)) 
                                   ])),
        batch_size=batch_size, shuffle=True)