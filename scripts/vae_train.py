import os

from utils.initialization_utils import set_default_workpath
from utils.train_test_util import train_util, train_steps

if __name__ == '__main__':
    set_default_workpath(os.path.dirname, os.getcwd())

    model, ema_model, train_loader, test_loader, optimizer, scheduler, epochs, run_path, device = (
        train_util("cfg/vae_s64_16x16x16.yaml"))

    train_steps(model, ema_model, train_loader, test_loader, optimizer, scheduler, epochs, run_path, device)
