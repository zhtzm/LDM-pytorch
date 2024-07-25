import os

import torch
from omegaconf import OmegaConf
from torch import optim
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from datasets.image_dataset import ImageDataset
from ldm.ldm import LDM
from ldm.vae import VAE


def get_ldm_from_cfg(yaml_path, device):
    config = OmegaConf.load(yaml_path)
    models_cfg = config["models"]
    cfg_dict = OmegaConf.to_container(models_cfg, resolve=True)

    vae = VAE(**cfg_dict['VAE']).to(device)
    ldm = LDM(ae_model=vae, condition_model=None, **cfg_dict['LDM']).to(device)

    return ldm


def get_dataloader(yaml_path):
    config = OmegaConf.load(yaml_path)
    dataset_cfg = OmegaConf.merge(config["dataset"], config["image"])

    transform = transforms.Compose([
        transforms.Resize((dataset_cfg.image_size, dataset_cfg.image_size)),
        transforms.ToTensor()
    ])
    dataset = ImageDataset(image_dir=dataset_cfg.data_dir, transform=transform)
    train_size = int(dataset_cfg.train_radio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=dataset_cfg.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=dataset_cfg.batch_size, shuffle=False)

    return train_loader, test_loader


def get_optimizer(yaml_path, model):
    config = OmegaConf.load(yaml_path)
    optim_cfg = config["optimizer"]
    optim_type = optim_cfg.type
    if optim_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.lr)
    elif optim_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=optim_cfg.lr)
    elif optim_type == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=optim_cfg.lr)
    else:
        optimizer = None

    return optimizer


def _generate_unique_filepath(base_path, state='train'):
    number = 0
    while True:
        run_path = f"{state}_{number}"
        run_path = os.path.join(base_path, run_path)
        if not os.path.exists(run_path):
            os.makedirs(run_path)
            return run_path
        number += 1


def save_checkpoint(epochs, model, ema, optimizer, run_path, filename):
    use_ema = True if ema is not None else False
    checkpoint = {
        'epochs': epochs,
        'use_ema': use_ema,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    if use_ema:
        checkpoint['ema_state_dict'] = ema.state_dict()

    torch.save(checkpoint, os.path.join(run_path, filename))


def load_checkpoint(ckpt_file, model, ema, optimizer):
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"Checkpoint file {ckpt_file} does not exist.")

    checkpoint = torch.load(ckpt_file)
    model.load_state_dict(checkpoint['model_state_dict'])

    if ema is not None and hasattr(ema, 'load_state_dict'):
        ema.load_state_dict(checkpoint['ema_state_dict'])
    else:
        print("EMA object does not have a load_state_dict method, EMA state not loaded.")

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['epochs'], checkpoint['use_ema']


def train_one_epoch(model, ema, optimizer, train_loader, device, epoch=None):
    model.train()
    acc_train_loss = 0

    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch} - Training")
    for x in train_loader_tqdm:
        x = x.to(device)
        loss = model(x)

        acc_train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None:
            ema.update(model)

        train_loader_tqdm.set_postfix(loss=loss.item())

    acc_train_loss /= len(train_loader)
    return acc_train_loss


def test_one_epoch(model, test_loader, device):
    model.eval()
    test_loss = 0
    test_loader_tqdm = tqdm(test_loader, desc="Testing")

    with torch.no_grad():
        for data in test_loader_tqdm:
            inputs = data.to(device)
            loss = model(inputs)
            test_loss += loss.item()

            test_loader_tqdm.set_postfix(loss=loss.item())

    test_loss /= len(test_loader)
    return test_loss


def train_model(model, ema, optimizer, train_loader, test_loader, **kwargs):
    epochs = kwargs['epochs']
    device = kwargs['device']
    run_path = kwargs['run_path']
    log_rate = kwargs['log_rate']
    save = kwargs['save']

    run_path = _generate_unique_filepath(run_path)
    model = model.to(device)
    ema = ema.to(device)

    best_test_loss = float('inf')
    for epoch in range(1, epochs + 1):
        _ = train_one_epoch(model, ema, optimizer, train_loader, device, epoch)

        if epoch % log_rate == 0:
            test_loss = test_one_epoch(model, test_loader, device)

            if test_loss < best_test_loss and save:
                best_test_loss = test_loss
                save_checkpoint(epoch, model, ema, optimizer, run_path, 'best_model.pth')

    if save:
        save_checkpoint(epochs, model, ema, optimizer, run_path, 'final_model.pth')
