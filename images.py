import torch
from torch import nn
from torchvision.io import write_jpeg
from torchvision import transforms


def generate_image(model: nn.Module, image_shape, output_path, latent_dim=3):
    to_pil = transforms.ToPILImage()
    to_tensor = transforms.ToTensor()

    image = torch.zeros(image_shape)
    for x in range(image_shape[0]):
        for y in range(image_shape[1]):
            X = torch.tensor([[float(x / image_shape[2]), float(y / image_shape[1])]])
            z = torch.zeros((X.shape[0], latent_dim))
            X = torch.cat([X, z], 1)
            out = model(X)
            image[0][y][x] = out[0][0]
            if latent_dim > 1:
                image[1][y][x] = out[0][1]
                image[2][y][x] = out[0][2]
            if latent_dim == 4:
                image[3][y][x] = out[0][3]

    image = to_pil(image)
    image = to_tensor(image)
    image *= 255
    image = image.to(torch.uint8)
    write_jpeg(image, output_path)
