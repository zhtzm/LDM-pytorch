import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

from ldm.vae import VAE


def disabled_train(self, mode=True):
    return self


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    embed_dim = 4
    coder_config = {'init_channels': 3,
                    'base_channels': 32,
                    'final_channels': 4,
                    'channel_mults': [1, 2, 2, 4],
                    'use_attention': [],
                    'num_res_blocks': 2,
                    'num_groups': 32,
                    'dropout': 0.1}

    pre_ae = torch.load("runs/best_ckpt.pth")
    vae = VAE(coder_config, embed_dim).to(device)
    vae.load_state_dict(pre_ae['model_state_dict'])
    vae = vae.eval()
    vae.train = disabled_train
    for param in vae.parameters():
        param.requires_grad = False

    img = Image.open('E:\\datasets\\animefaces-konachan\\000000-01.jpg').convert("RGB")
    img = transform(img)

    image = img.cpu().permute(1, 2, 0).numpy()  # 将CHW格式转换为HWC，并转换为numpy数组
    image_ = (image * 255).astype(np.uint8)
    plt.imshow(image_)
    plt.axis('off')  # 不显示坐标轴
    plt.show()

    x = img.reshape(1, 3, 256, 256)
    x, _, _ = vae(x.to(device))

    x = x[0]
    image = x.cpu().permute(1, 2, 0).numpy()  # 将CHW格式转换为HWC，并转换为numpy数组
    image_ = (image * 255).astype(np.uint8)
    plt.imshow(image_)
    plt.axis('off')  # 不显示坐标轴
    plt.show()
