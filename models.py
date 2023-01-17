import torch
from collections import OrderedDict
from torch import nn
from torchvision.io import write_jpeg
from torchvision import transforms
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

    def generate_image(self, image_shape, output_path, latent_dim=3):
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        image = torch.zeros(image_shape)
        for x in range(image_shape[1]):
            for y in range(image_shape[2]):
                X = torch.tensor([[float(x / image_shape[2]), float(y / image_shape[1])]])

                if latent_dim > 0:
                    z = torch.zeros((X.shape[0], latent_dim))
                    X = torch.cat([X, z], 1)

                out = self.forward(X)
                image[0][y][x] = out[0][0]
                if image_shape[0] > 1:
                    image[1][y][x] = out[0][1]
                    image[2][y][x] = out[0][2]
                if image_shape[0] == 4:
                    image[3][y][x] = out[0][3]

        image = to_pil(image)
        image = to_tensor(image)
        image *= 255
        image = image.to(torch.uint8)
        write_jpeg(image, output_path)


if __name__ == '__main__':
    model = CPPN(num_layers=9, num_nodes=32).to('cpu')
    model.generate_image((3, 64, 64), 'test.jpg', latent_dim=1)

    # generate_image(model, (3, 64, 64), 'test.jpg', latent_dim=1)
