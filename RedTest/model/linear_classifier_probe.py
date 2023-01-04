import torch
import torch.nn as nn
import torch.nn.functional as F


class Dynamic_linear_layer(nn.Module):
    def __init__(self, in_planes, num_classes):
        super(Dynamic_linear_layer, self).__init__()
        self.fc1 = nn.Linear(in_planes, num_classes)

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        return out


