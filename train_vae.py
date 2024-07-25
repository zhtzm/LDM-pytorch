import os

import torch
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms
from tqdm import tqdm

from datasets.image_dataset import ImageDataset
from ldm.vae import VAE

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

    embed_dim = 4
    coder_config = {'init_channels': 3,
                    'base_channels': 32,
                    'final_channels': 4,
                    'channel_mults': [1, 2, 4],
                    'use_attention': [],
                    'num_res_blocks': 2,
                    'num_groups': 32,
                    'dropout': 0.1}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = VAE(coder_config, embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.00005)

    epochs = 400

    losses = {}
    best_test_loss = float('inf')
    for epoch in range(1, epochs + 1):
        model.train()
        acc_train_loss = 0

        epoch_loss = {}

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch} - Training")
        for x in train_loader_tqdm:
            x = x.to(device)
            dec, posterior = model(x)
            loss = torch.nn.functional.l1_loss(dec, x)

            acc_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loader_tqdm.set_postfix(loss=loss.item())

        acc_train_loss /= len(train_loader)
        epoch_loss['train'] = acc_train_loss

        model.eval()
        test_loss = 0
        test_loader_tqdm = tqdm(test_loader, desc="Testing")

        with torch.no_grad():
            for data in test_loader_tqdm:
                inputs = data.to(device)
                dec, posterior = model(inputs)
                loss = torch.nn.functional.l1_loss(inputs, dec)
                test_loss += loss.item()

                test_loader_tqdm.set_postfix(loss=loss.item())

        test_loss /= len(test_loader)
        epoch_loss['test'] = acc_train_loss

        if epoch % 10 == 0:
            use_ema = False
            checkpoint = {
                'epochs': epochs,
                'use_ema': use_ema,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }

            torch.save(checkpoint, os.path.join('runs/size128/VAE', f'ckpt_{epoch}.pth'))

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(checkpoint, os.path.join('runs/size128/VAE', 'best_ckpt.pth'))

        losses[epoch] = epoch_loss

    use_ema = False
    checkpoint = {
        'epochs': epochs,
        'use_ema': use_ema,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, os.path.join('runs/size128/VAE', 'final.pth'))

    # 提取训练和测试损失
    train_losses = [loss['train'] for epoch, loss in losses.items()]
    test_losses = [loss['test'] for epoch, loss in losses.items()]

    # 提取epoch编号
    epochs = list(losses.keys())

    # 绘制训练和测试损失
    plt.figure(figsize=(10, 5))  # 设置图形大小
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, test_losses, label='Test Loss', marker='x')

    # 添加标题和标签
    plt.title('Training and Test Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 显示图例
    plt.legend()

    # 显示网格
    plt.grid(True)

    # 显示图形
    plt.show()

