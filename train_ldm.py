import os

import torch
from torch import optim
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from datasets.image_dataset import ImageDataset
from ldm.ldm import LDM
from ldm.vae import VAE


def disabled_train(self, mode=True):
    return self


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    dataset = ImageDataset(image_dir="E:\\datasets\\animefaces-konachan", transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embed_dim = 4
    coder_config = {'init_channels': 3,
                    'base_channels': 32,
                    'final_channels': 4,
                    'channel_mults': [1, 2, 2],
                    'use_attention': [],
                    'num_res_blocks': 1,
                    'num_groups': 32,
                    'dropout': 0.1}

    pre_ae = torch.load("runs/VAE/final_model.pth")
    vae = VAE(coder_config, embed_dim).to(device)
    vae.load_state_dict(pre_ae['model_state_dict'])
    vae = vae.eval()
    vae.train = disabled_train
    for param in vae.parameters():
        param.requires_grad = False

    LDM_cfg = {'conditioning_key': None, 'ddpm_config': {'beta': [0.00085, 0.012],
                                                         't_max': 500,
                                                         'channel': 4,
                                                         'size': 32,
                                                         'clip_denoised': True,
                                                         'sample_img_every_t': 100,
                                                         'loss_type': 'l1',
                                                         'unet_config': {
                                                             't_max': 500,
                                                             'init_channels': 4,
                                                             'base_channels': 128,
                                                             'channel_mults': [1, 2, 4, 4],
                                                             'use_attention': [2],
                                                             'num_res_blocks': 2,
                                                             'num_groups': 32,
                                                             'dropout': 0.1,
                                                             'time_emb_scale': 1.0,
                                                             'use_t_emb': True
                                                         }
                                                         }
               }

    model = LDM(ae_model=vae, condition_model=None, **LDM_cfg).to(device=device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    epochs = 200

    best_test_loss = float('inf')
    for epoch in range(1, epochs + 1):
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

            train_loader_tqdm.set_postfix(loss=loss.item())

        acc_train_loss /= len(train_loader)

        if epoch % 20 == 0:
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

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                use_ema = False
                checkpoint = {
                    'epochs': epochs,
                    'use_ema': use_ema,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }

                torch.save(checkpoint, os.path.join('runs/LDM', 'best_model.pth'))

    use_ema = False
    checkpoint = {
        'epochs': epochs,
        'use_ema': use_ema,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, os.path.join('runs/LDM', 'final_model.pth'))
