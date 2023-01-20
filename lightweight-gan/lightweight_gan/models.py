import torch
from collections import OrderedDict
from torch import nn
from torchvision.io import write_jpeg
from torchvision import transforms


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

    @property
    def device(self):
        return next(self.parameters()).device

    """
    When output_path is None, the image is returned as a tensor.
    """
    def generate_image(self, image_shape, output_path=None, latent_dim=3, verbose=False, latent_vector=None):
        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        image = torch.zeros(image_shape)
        step = 0
        for x in range(image_shape[1]):
            for y in range(image_shape[2]):
                if verbose:
                    print(f"Image Progress: {step / (image_shape[1] * image_shape[2]) * 100:.2f}%", end="\r")
                    step += 1

                X = torch.tensor([[float(x / image_shape[2]), float(y / image_shape[1])]])

                if latent_vector is not None:
                    X = X.repeat(latent_vector.shape[0], 1)
                    X = X.to(self.device)

                if latent_dim > 0:
                    z = torch.zeros((X.shape[0], latent_dim)) if latent_vector is None else latent_vector
                    X = torch.cat([X, z], 1)

                X = X.to(torch.float32)
                X = X.to(self.device)

                out = self.forward(X)
                image[0][y][x] = out[0][0]
                if image_shape[0] > 1:
                    image[1][y][x] = out[0][1]
                    image[2][y][x] = out[0][2]
                if image_shape[0] == 4:
                    image[3][y][x] = out[0][3]

        image = to_pil(image)
        image = to_tensor(image)

        if output_path is None:
            return image

        image *= 255
        image = image.to(torch.uint8)
        write_jpeg(image, output_path)


if __name__ == '__main__':
    model = CPPN(num_layers=9, num_nodes=32).to('cpu')
    out = model.generate_image((3, 64, 64), latent_dim=1, verbose=True, latent_vector=torch.tensor([[0.5]]))
    print(out.shape)