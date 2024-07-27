import numpy as np
import torch
from matplotlib import pyplot as plt

from ldm.ldm import LDM
from ldm.vae import VAE


def disabled_train(self, mode=True):
    return self


if __name__ == '__main__':
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

    pre_ae = torch.load("runs/VAE/best_model.pth")
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

    pre_ldm = torch.load("runs/LDM/best_model.pth")
    ldm = LDM(ae_model=vae, condition_model=None, **LDM_cfg).to(device=device)
    ldm.load_state_dict(pre_ldm['model_state_dict'])
    ldm = ldm.eval()
    ldm.train = disabled_train
    for param in ldm.parameters():
        param.requires_grad = False

    imgs, intermediates = ldm.sample(2, True)

    b, c, h, w = intermediates[0].shape

    # 遍历每个Tensor和它的子图
    for img in imgs:
        # 确保Tensor是CPU上的，并且数据类型适合显示
        image = img.cpu().permute(1, 2, 0).numpy()  # 将CHW格式转换为HWC，并转换为numpy数组

        image_ = (image * 255).astype(np.uint8)

        plt.imshow(image_)
        plt.axis('off')  # 不显示坐标轴
        plt.show()
