import os

import torch
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from tqdm import tqdm

from utils.initialization_utils import parse_yaml_config, initialize_model, import_class_from_string, \
    generate_unique_filepath, initialize_dataloader


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


def train_util(config_path: str):
    config = parse_yaml_config(config_path)

    model_cfg = config.get("models")
    assert model_cfg is not None, "models should not be None"
    main_model, class_name, ema_model = initialize_model(model_cfg)

    dataset_cfg = config.get("dataset")
    assert dataset_cfg is not None, "dataset should not be None"
    train_loader, test_loader = initialize_dataloader(dataset_cfg)

    train_cfg = config.get("train")
    assert train_cfg is not None, "train should not be None"
    epochs = train_cfg['epochs']
    device = torch.device(train_cfg['device'] if torch.cuda.is_available() else 'cpu')
    run_path = generate_unique_filepath(base_path=train_cfg['base_path'], class_name=class_name, state='train')

    optimizer_class, _ = import_class_from_string(train_cfg['optimizer']['target'])
    optimizer = optimizer_class(main_model.parameters(), **train_cfg['optimizer']['params'])

    scheduler_class, _ = import_class_from_string(train_cfg['scheduler']['target'])
    scheduler = scheduler_class(optimizer, **train_cfg['scheduler']['params'])

    return main_model, ema_model, train_loader, test_loader, optimizer, scheduler, epochs, run_path, device


def test_util(config_path: str, checkpoint_path: str):
    config = parse_yaml_config(config_path)

    model_cfg = config.get("models")
    assert model_cfg is not None, "models should not be None"
    main_model, class_name, _ = initialize_model(model_cfg)

    ckpt = torch.load(checkpoint_path)
    main_model.load_state_dict(ckpt['model_state_dict'])
    main_model = main_model.eval()

    dataset_cfg = config.get("dataset")
    assert dataset_cfg is not None, "dataset should not be None"
    transform = transforms.Compose([
        transforms.Resize((dataset_cfg['image_size'], dataset_cfg['image_size'])),
        transforms.ToTensor()
    ])

    test_cfg = config.get("test")
    assert test_cfg is not None, "train should not be None"
    device = torch.device(test_cfg['device'] if torch.cuda.is_available() else 'cpu')

    return main_model, device, transform


def draw_loss(losses: dict, run_path: str):
    plt.plot(losses['train'], label='Train Loss')
    plt.plot(losses['test'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_path, 'loss_over_epochs.png'))


def train_steps(model, ema_model, train_loader, test_loader, optimizer, scheduler, epochs, run_path, device):
    model.to(device)
    if ema_model is not None:
        ema_model.to(device)

    losses = {'train': [], 'test': []}
    best_test_loss = float('inf')
    show_best_test_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        acc_train_loss = 0
        total_samples = 0
        lr = optimizer.param_groups[0]['lr']

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} - Training")
        for x in train_loader_tqdm:
            x = x.to(device)
            loss = model.loss_forward(x)

            acc_train_loss += loss['loss'].item()
            total_samples += x.size(0)
            optimizer.zero_grad()
            loss['loss'].backward()
            optimizer.step()
            if ema_model is not None:
                ema_model.update(model)

            train_loader_tqdm.set_postfix(loss=acc_train_loss/total_samples, lr=lr)

        acc_train_loss /= total_samples
        losses['train'].append(acc_train_loss)

        model.eval()
        test_loss = 0
        total_samples = 0
        test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {epoch}/{epochs} - Testing")

        with torch.no_grad():
            for data in test_loader_tqdm:
                inputs = data.to(device)
                loss = model.loss_forward(inputs)
                test_loss += loss['loss'].item()
                total_samples += inputs.size(0)

                test_loader_tqdm.set_postfix(loss=test_loss/total_samples, sbtl=show_best_test_loss)

        test_loss /= total_samples
        losses['test'].append(test_loss)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(test_loss)
        else:
            scheduler.step()

        if test_loss < show_best_test_loss:
            show_best_test_loss = test_loss

        if epoch % 10 == 0 or epoch == epochs:
            state = {
                'epochs': epochs,
                'use_ema': False if ema_model is None else True,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_model_state_dict': None if ema_model is None else ema_model.state_dict(),
            }
            save_checkpoint(state=state, is_best=False, checkpoint_dir=run_path, filename=f'ckpt_{epoch}.pth')
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                save_checkpoint(state=state, is_best=True, checkpoint_dir=run_path)

    draw_loss(losses=losses, run_path=run_path)
