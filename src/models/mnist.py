from torch import nn
import torch.nn.functional as F


class MnistExampleV01(nn.Module):
    """
    Simple convolution neural network originally designed for mnist but modified for APTOS dataset.

    Original:
    https://github.com/pytorch/examples/blob/master/mnist/main.py
    """
    def __init__(self, shape):
        super().__init__()
        self.conv1 = nn.Conv2d(shape[0], 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(50 * 22 * 22, 500)
        self.fc2 = nn.Linear(500, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 50 * 22 * 22)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
