import torch
from torch import nn
from typing import Tuple


class ResidualBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(ResidualBlock, self).__init__()
        self.create_residual_connection = True if input_dim == output_dim else False
        self.activation = nn.ReLU()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        y = self.activation(self.linear(x))
        if self.create_residual_connection:
            y = x + y
        return y


class ResidualNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int], flatten: bool = False):
        super(ResidualNN, self).__init__()
        blocks = list()
        self.input_dim = input_dim
        self.flatten = flatten
        block_input_dim = input_dim
        for hidden_dim in hidden_dims:
            blocks.append(ResidualBlock(block_input_dim, hidden_dim))
            block_input_dim = hidden_dim
        blocks.append(nn.Linear(block_input_dim, output_dim))
        self.network = nn.Sequential(*blocks)

    def forward(self, x):
        if self.flatten:
            x = x.view(x.shape[0], -1)
        output = self.network(x)
        return output


class FeedForwardNN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int]):
        super().__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(input_dim, hidden_dim), nn.PReLU()]
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim

    def forward(self, *args):
        return self.network(torch.cat(args, dim=-1))
