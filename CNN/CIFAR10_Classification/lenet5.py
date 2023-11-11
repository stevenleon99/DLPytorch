import torch
from torch import _nnpack_available
import torch.nn as nn
from torch import functional as F

class Letnet5(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Letnet5, self).__init__(*args, **kwargs)
        self.conv_unit = nn.Sequential(
            # x: [32, 3, 32, 32]
            nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=0),
            # x: [32, 6, H, W]
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            # x: [32, 6, H2, W2]
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )
    
        #fc unit
        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
    
    def forward(self, x):
        """
        params: x [b, 3, 32, 32]
        """
        batch_size = x.size(0)
        x = self.conv_unit(x)
        x = x.view(batch_size, 16*5*5)
        logits = self.fc_unit(x)
        
        # pred = F.softmax(logits, dim=1)
        # loss = self.criteon(logits, y)
        return logits


if __name__ == "__main__":
    # net = Letnet5()
    # input = torch.rand(32,3,32,32)
    # print(net.conv_unit(input).shape) #out [32, 16, 5,5]
    # out = net.forward(input)
    # print(out.shape)

    # label = torch.full([10], 1)
    # logits = torch.rand((10, 10))
    # criteon = nn.SoftMarginLoss()
    # loss = criteon(logits, label)
    # print(loss)
    pass