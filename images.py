import numpy as np
import torch
from torch import nn
from torchvision.io import write_jpeg
from torchvision import transforms


def positional_encode(x, y, bins):
    vals = []
    vals.append(x)
    vals.append(y)
    for i in range(bins):
        vals.append(np.sin(x * 2 * np.pi * 2 ** i))
        vals.append(np.cos(x * 2 * np.pi * 2 ** i))
        vals.append(np.sin(y * 2 * np.pi * 2 ** i))
        vals.append(np.cos(y * 2 * np.pi * 2 ** i))

    return torch.tensor([vals])


def generate_image(model: nn.Module, image_shape, output_path, latent_dim=3, positional_encoding_bins=12, output_dim=3, device="cpu"):
    model.to(device)

    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    image = torch.zeros(image_shape)
    for x in range(image_shape[1]):
        for y in range(image_shape[2]):
            X = positional_encode(x/image_shape[0], y/image_shape[1], positional_encoding_bins)

            if latent_dim > 0:
                z = torch.zeros((X.shape[0], latent_dim))
                X = torch.cat([X, z], 1)

            X = X.to(torch.float32)
            X = X.to(device)

            out = model(X)
            image[0][x][y] = out[0][0]
            if output_dim > 1:
                image[1][x][y] = out[0][1]
                image[2][x][y] = out[0][2]
            if output_dim == 4:
                image[3][x][y] = out[0][3]

    image = to_pil(image)
    image = to_tensor(image)
    image *= 255
    image = image.to(torch.uint8)
    write_jpeg(image, output_path)
