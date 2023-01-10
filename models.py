from typing import Optional, Union, overload

import torch
from collections import OrderedDict
from torch import nn, device, dtype, Tensor

# from train import generate_image
from torch.nn.modules.module import T


def dense(i, o):
    layer = nn.Linear(i, o)
    layer.weight.data = torch.normal(0, 1, (o, i))
    layer.bias.data = torch.normal(0, 1, (o,))
    return layer


"""
    Input vector should be at least 12 dimensions long + z-length
"""


class CPPN(nn.Module):
    def __init__(self, input_vector_length=12, num_layers=6, num_nodes=16, output_vector_length=3,
                 positional_encoding_bins=12):
        super().__init__()

        self.positional_encoding_bins = positional_encoding_bins

        layers = [
            ('input_layer', dense((positional_encoding_bins * 4) + 2 + input_vector_length, num_nodes))
        ]

        for i in range(num_layers):
            layers.append(("tan_{}".format(i), nn.Tanh()))
            layers.append(("dense_{}".format(i), dense(num_nodes, num_nodes)))

        layers.append(("final_tanh", nn.Tanh()))
        layers.append(("final_layer", dense(num_nodes, output_vector_length)))
        layers.append(("sigmoid", nn.Sigmoid()))

        self.network = nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        return self.network(x)

    @overload
    def to(self: T, device: Optional[Union[int, device]] = ..., dtype: Optional[Union[dtype, str]] = ...,
           non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T, dtype: Union[dtype, str], non_blocking: bool = ...) -> T:
        ...

    @overload
    def to(self: T, tensor: Tensor, non_blocking: bool = ...) -> T:
        ...

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.network = self.network.to(*args, **kwargs)
        return self

    @staticmethod
    def positional_encode(x, y, bins):
        x = x * bins
        y = y * bins
        x = int(x)
        y = int(y)
        encoding = torch.zeros(bins * 4)
        encoding[x] = 1
        encoding[bins + y] = 1
        encoding[bins * 2 + x + y] = 1
        encoding[bins * 3 + x - y] = 1
        return encoding


if __name__ == '__main__':
    model = CPPN(num_layers=9, num_nodes=32).to('cpu')
    print(model)

    for param in model.named_parameters():
        print(param)

    # generate_image(model, (3, 64, 64), 'test.jpg', latent_dim=1)
