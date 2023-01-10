import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision import transforms
from torchvision.io import read_image


class SingleImageDataset(Dataset):
    def __init__(self, image_path, alpha=False, positional_encoding_bins=12):
        super().__init__()

        to_pil = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()

        self.image = read_image(image_path)
        self.image = to_pil(self.image)
        self.image = to_tensor(self.image)
        self.alpha = alpha
        self.positional_encoding_bins = positional_encoding_bins

    def __len__(self):
        return self.image.shape[1] * self.image.shape[2]

    def __getitem__(self, index) -> T_co:
        height = self.image.shape[1]
        width = self.image.shape[2]

        x = index // width
        y = index % width

        r = self.image[0, y, x]
        g = self.image[1, y, x]
        b = self.image[2, y, x]
        a = self.image[3, y, x] if self.alpha else 1.0

        x = x / width
        y = y / height

        x_vals = []
        x_vals.append(x)
        x_vals.append(y)
        for i in range(self.positional_encoding_bins):
            x_vals.append(np.sin(x * 2 * np.pi * 2 ** i))
            x_vals.append(np.cos(x * 2 * np.pi * 2 ** i))
            x_vals.append(np.sin(y * 2 * np.pi * 2 ** i))
            x_vals.append(np.cos(y * 2 * np.pi * 2 ** i))

        return np.array(x_vals), np.array([r, g, b, a] if self.alpha else [r, g, b])


if __name__ == '__main__':
    test_dataset = SingleImageDataset('/Users/allanpichardo/sites/cppn-generator/test_image.png')
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    train_features, train_labels = next(iter(test_dataloader))
    print(f"Feature batch shape: {train_features.shape}")
    print(f"Labels batch shape: {train_labels.shape}")

    for x, y in zip(train_features, train_labels):
        print("Pixel {} | Color {},{},{}".format(x, y[0], y[1], y[2]))
