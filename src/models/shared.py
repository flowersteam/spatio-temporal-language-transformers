import torch
import torch.nn as nn

class Feedforward(torch.nn.Module):
    " A simple multilayer fully connected network"

    def __init__(self, input_size, layers_sizes):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.layers_sizes = layers_sizes
        self.fc_layers = nn.ModuleList()
        self.activations = []

        size_tmp = self.input_size
        for i, size in enumerate(self.layers_sizes):
            self.activations.append(nn.ReLU() if i < len(self.layers_sizes) - 1 else None)
            fc = torch.nn.Linear(size_tmp, size)
            nn.init.kaiming_uniform_(fc.weight)
            nn.init.zeros_(fc.bias)
            self.fc_layers.append(fc)
            size_tmp = size

    def forward(self, input):
        for fc, activation in zip(self.fc_layers, self.activations):
            input = fc(input)
            if activation:
                input = activation(input)
        return input