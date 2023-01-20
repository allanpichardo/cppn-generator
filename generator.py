from lightweight_gan import Generator


if __name__ == '__main__':
    generator = Generator(image_size=128)
    generator.cppn.generate_image((3, 128, 128), latent_dim=256,output_path="test_image.png", verbose=True)