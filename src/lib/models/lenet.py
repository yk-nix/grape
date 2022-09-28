from torch.nn import Module
from torch import Tensor
import torch.nn as nn

class LeNet(Module):
    """
    input image: 1 x 28 x 28
    """
    name = 'lenet'
    
    def __init__(self, 
                 num_class : int = 10) -> None:
        super(LeNet, self).__init__()
        self.num_class = num_class
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5)
        self.fc1   = nn.Linear(in_features = 16 * 4 * 4, out_features = 128)
        self.fc2   = nn.Linear(in_features = 128, out_features = 56)
        self.fc3   = nn.Linear(in_features = 56, out_features = num_class)
        self.mp    = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.relu  = nn.ReLU(inplace=True)
        
    def forward(self, 
                x : Tensor) -> Tensor:
        """
        input: b x 1 x 28 x 28   (batch x channel x height x width)
        ouput: b x 10            (batch x num_class)
        """
        b = x.shape[0]
        x = self.mp(self.conv1(x))
        x = self.mp(self.conv2(x))
        x = x.reshape(b, -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.relu(self.fc3(x))