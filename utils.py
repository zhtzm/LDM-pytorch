import os

import torch
from torch.utils.data import random_split, DataLoader

from datasets.image_dataset import ImageDataset


def initialize_model(model_class, model_config: dict, device: torch.device):
    model = model_class(**model_config).to(device)
    return model


def load_datasets(data_dir, transform, split_ratio=0.8, batch_size=32):
    dataset = ImageDataset(data_dir, transform=transform)
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size // 2, shuffle=False)
    return train_loader, test_loader


def _generate_unique_filepath(base_path, state='train'):
    number = 0
    while True:
        run_path = f"{state}_{number}"
        run_path = os.path.join(base_path, run_path)
        if not os.path.exists(run_path):
            os.makedirs(run_path)
            return run_path
        number += 1


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(state, os.path.join(checkpoint_dir, filename))
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, 'best_checkpoint.pth'))


def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer
