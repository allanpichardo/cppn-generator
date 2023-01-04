import torch
from collections import OrderedDict
from torch import nn
# from train import generate_image


def dense(i, o):
    layer = nn.Linear(i, o)
    layer.weight.data = torch.normal(0, 1, (o, i))
    layer.bias.data = torch.normal(0, 1, (o,))
    return layer


class CPPN(nn.Module):
    def __init__(self, input_vector_length=3, num_layers=6, num_nodes=16, output_vector_length=3):
        super().__init__()

        layers = [
            ('input_layer', dense(input_vector_length, num_nodes))
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


if __name__ == '__main__':
    model = CPPN(num_layers=9, num_nodes=32).to('cpu')
    print(model)

    for param in model.named_parameters():
        print(param)

    # generate_image(model, (3, 64, 64), 'test.jpg', latent_dim=1)
