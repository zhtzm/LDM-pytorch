import os

from utils.initialization_utils import set_default_workpath
from utils.train_test_util import train_util, train_steps

if __name__ == '__main__':
    set_default_workpath(os.path.dirname, os.getcwd())

    model, ema_model, train_loader, test_loader, optimizer, scheduler, epochs, run_path, device = \
        train_util("cfg/ldm_32x32x4_8x8_nc.yaml")

    train_steps(model, ema_model, train_loader, test_loader, optimizer, scheduler, epochs, run_path, device)
