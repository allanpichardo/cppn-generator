import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from images import generate_image
from datasets import SingleImageDataset
from models import CPPN


def train(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer: torch.optim.Optimizer,
          model_save_path=os.path.join(os.path.dirname(__file__), "model.pth"), epochs=100, device="cpu", latent_dim=3,
          channels=3, positional_encoding_bins=12):
    if os.path.exists(model_save_path):
        print("Found existing model, loading...")
        print("Loading model from {}".format(model_save_path))
        model.load_state_dict(torch.load(model_save_path))

    if not os.path.exists(os.path.join(os.path.dirname(model_save_path), 'images')):
        os.makedirs(os.path.join(os.path.dirname(model_save_path), 'images'))

    model.to(device)

    generate_image(model, (channels, 128, 128),
                   os.path.join(os.path.dirname(__file__), 'images', f"epoch-{0}.png"),
                   latent_dim=latent_dim, positional_encoding_bins=positional_encoding_bins, output_dim=channels, device=device)

    try:
        for epoch in range(epochs):
            print("Epoch {}/{}".format(epoch + 1, epochs))
            print("-" * 10)
            size = len(dataloader.dataset)
            print(f"{size} total batches")
            model.train()
            for batch, (X, y) in enumerate(dataloader):
                if latent_dim > 0:
                    z = torch.zeros((X.shape[0], latent_dim))
                    X = torch.cat([X, z], 1)

                X = X.to(torch.float32)
                y = y.to(torch.float32)

                X, y = X.to(device), y.to(device)

                pred = model(X)
                loss = loss_fn(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch % 100 == 0:
                    loss, current = loss.item(), batch * len(X)
                    print(f"loss: {loss:>7f}  [{(current / size * 100):.2f}%]")

            print("Epoch finished. Saving image...")
            generate_image(model, (channels, 128, 128),
                           os.path.join(os.path.dirname(__file__), 'images', f"epoch-{epoch + 1}.png"),
                           latent_dim=latent_dim, positional_encoding_bins=positional_encoding_bins, output_dim=channels, device=device)

    except KeyboardInterrupt:
        print("Training stopped early")
        torch.save(model.state_dict(), model_save_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a CPPN')
    parser.add_argument('--model', type=str, default='model.pth', help='Path to save the model to')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device to train on')
    parser.add_argument('--image_path', type=str, default='test_image.png', help='Image to train on')
    parser.add_argument('--model_width', type=int, default=25, help='Width of the model')
    parser.add_argument('--model_depth', type=int, default=9, help='Depth of the model')
    parser.add_argument('--latent_dim', type=int, default=12, help='Latent dimension of the model')
    parser.add_argument('--output_dim', type=int, default=3, help='Output dimension of the model')
    parser.add_argument('--positional_encoding_bins', type=int, default=12, help='Number of positional encoding bins')
    args = parser.parse_args()

    model = CPPN(input_vector_length=args.latent_dim, num_layers=args.model_depth, num_nodes=args.model_width,
                 output_vector_length=args.output_dim, positional_encoding_bins=args.positional_encoding_bins)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    dataset = SingleImageDataset(args.image_path, alpha=args.output_dim == 4)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    train(dataloader, model, loss_fn, optimizer, model_save_path=args.model, epochs=args.epochs, device=args.device,
          latent_dim=args.latent_dim, channels=args.output_dim, positional_encoding_bins=model.positional_encoding_bins)
    print("Training finished")
    torch.save(model.state_dict(), args.model)
