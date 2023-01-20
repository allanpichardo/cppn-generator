from lightweight_gan import Generator
import torch


if __name__ == '__main__':
    generator = Generator(image_size=128)
    generator.eval()

    z = torch.randn(8, 256)
    x = generator(z)

    print(x.shape)