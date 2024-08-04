import os

import torch
from tqdm import tqdm

from utils.path_utils import set_default_workpath
from utils.utils import train_util, save_checkpoint, draw_loss

if __name__ == '__main__':
    set_default_workpath(os.path.dirname, os.getcwd())

    model, ema_model, optimizer, train_loader, test_loader, epochs, run_path, device = (
        train_util("cfg/vae_s64_16x16x16.yaml"))

    losses = {}
    best_test_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        acc_train_loss = 0

        epoch_loss = {}

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs + 1} - Training")
        for x in train_loader_tqdm:
            x = x.to(device)
            recon, mean, log_var = model(x)
            loss = model.loss_function(x, recon, mean, log_var, kl_weight=0.5)

            acc_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ema_model is not None:
                ema_model.update(model)

            train_loader_tqdm.set_postfix(loss=loss.item())

        acc_train_loss /= len(train_loader)
        epoch_loss['train'] = acc_train_loss

        model.eval()
        test_loss = 0
        test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {epoch}/{epochs + 1} - Testing")

        with torch.no_grad():
            for data in test_loader_tqdm:
                inputs = data.to(device)
                recon, mean, log_var = model(inputs)
                loss = model.loss_function(inputs, recon, mean, log_var, kl_weight=0.005)
                test_loss += loss.item()

                test_loader_tqdm.set_postfix(loss=loss.item())

        test_loss /= len(test_loader)
        epoch_loss['test'] = test_loss

        if epoch % 10 == 0 or epoch == epochs:
            state = {
                'epochs': epochs,
                'use_ema': False if ema_model is None else True,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'ema_model_state_dict': None if ema_model is None else ema_model.state_dict(),
            }
            save_checkpoint(state=state, is_best=False, checkpoint_dir=run_path, filename=f'ckpt_{epoch}.pth')

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                save_checkpoint(state=state, is_best=True, checkpoint_dir=run_path)

        losses[epoch] = epoch_loss

    draw_loss(losses=losses, run_path=run_path)
