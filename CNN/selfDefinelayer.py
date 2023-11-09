from torch import nn

class Flatten(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Flatten, self).__init__(*args, **kwargs)
    
    def forward(self, input):
        return input.view(input.size(0), -1) #can be used in nn.Squential()