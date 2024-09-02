import sys

from torch.utils.data import DataLoader
from torchinfo import summary

from utils.train_test_util import train_util, train_steps


def default_train_script(cfg_file_path):
    model, ema_model, train_loader, test_loader, optimizer, scheduler, epochs, run_path, device = (
        train_util(cfg_file_path))

    if isinstance(train_loader, DataLoader):
        first_batch = next(iter(train_loader))
        input_size = tuple(first_batch.size())
        print(f"Detected input size from train_loader: {input_size}")
        summary(model, input_size=input_size, depth=5)
    else:
        print("train_loader is not an instance of DataLoader.")

    train_steps(model, ema_model, train_loader, test_loader, optimizer, scheduler, epochs, run_path, device)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        cfg_file = sys.argv[1]
        default_train_script(cfg_file)
    else:
        print("Usage: python script.py <config_file_path>")
