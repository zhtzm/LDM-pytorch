import importlib
import os
import torch
import yaml
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

from models.ema import EMA


def parse_yaml_config(config_path: str):
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return config['models'], config['images'], config['dataset'], config['train']
    except FileNotFoundError:
        raise IOError(f"Configuration file {config_path} does not exist")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file {config_path}: {e}")


def import_class_from_string(class_string: str):
    module_name, class_name = class_string.rsplit('.', 1)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name), class_name
    except ImportError as e:
        raise ImportError(f"Could not import {class_string}. Reason: {e}")
    except AttributeError:
        raise AttributeError(f"Class {class_name} not found in module {module_name}.")


def disable_train_mode(model: nn.Module):
    model.eval()
    model.train = model.eval
    for param in model.parameters():
        param.requires_grad = False
    return model


def generate_unique_filepath(base_path, class_name, state='train'):
    number = 0
    while True:
        run_path = f"{class_name}_{state}_{number}"
        run_path = os.path.join(base_path, run_path)
        if not os.path.exists(run_path):
            os.makedirs(run_path)
            return run_path
        number += 1


def load_datasets(transform, dataset_class, data_dir, train_ratio=0.8, batch_size=32):
    dataset = dataset_class(data_dir, transform=transform)
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size // 2, shuffle=False)
    return train_loader, test_loader


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth'):
    if is_best:
        torch.save(state, os.path.join(checkpoint_dir, 'best_checkpoint.pth'))
    else:
        torch.save(state, os.path.join(checkpoint_dir, filename))


def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer


def initialize_main_model(model_cfg):
    main_model_class, class_name = import_class_from_string(model_cfg['main_model']['target'])
    return main_model_class(**model_cfg['main_model']['params']), class_name


def train_util(config_path: str):
    model_cfg, image_cfg, dataset_cfg, train_cfg = parse_yaml_config(config_path)

    main_model, class_name = initialize_main_model(model_cfg)

    dataset_class, _ = import_class_from_string(dataset_cfg['target'])
    transform = transforms.Compose([
        transforms.Resize((image_cfg['image_size'], image_cfg['image_size'])),
        transforms.ToTensor()
    ])
    train_loader, test_loader = load_datasets(transform=transform, dataset_class=dataset_class,
                                              **dataset_cfg['params'])

    use_ema = train_cfg['use_ema']
    epochs = train_cfg['epochs']
    device = torch.device(train_cfg['device'] if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(main_model.parameters(), lr=train_cfg['lr'])
    base_path = train_cfg['base_path']
    run_path = generate_unique_filepath(base_path=base_path, class_name=class_name, state='train')

    main_model = main_model.to(device)

    ema_model = EMA(main_model) if use_ema else None

    return main_model, ema_model, optimizer, train_loader, test_loader, epochs, run_path, device


def draw_loss(losses: dict, run_path: str):
    train_losses = [loss['train'] for _, loss in losses.items()]
    test_losses = [loss['test'] for _, loss in losses.items()]

    epochs = list(losses.keys())

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, test_losses, label='Test Loss', marker='x')

    plt.title('Training and Test Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_path, 'loss_over_epochs.png'))
