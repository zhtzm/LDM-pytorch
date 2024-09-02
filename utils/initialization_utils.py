import importlib
import os

import yaml
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


def import_class_from_string(class_string: str):
    module_name, class_name = class_string.rsplit('.', 1)
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name), class_name
    except ImportError as e:
        raise ImportError(f"Could not import {class_string}. Reason: {e}")
    except AttributeError:
        raise AttributeError(f"Class {class_name} not found in module {module_name}.")


def parse_yaml_config(config_path: str):
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            return config
    except FileNotFoundError:
        raise IOError(f"Configuration file {config_path} does not exist")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file {config_path}: {e}")


def initialize_model(model_cfg):
    main_model_cfg = model_cfg.get("main_model")
    assert main_model_cfg is not None, "main_model should not be None"
    main_model_class, class_name = import_class_from_string(main_model_cfg['target'])
    main_model = main_model_class(**main_model_cfg['params'])

    ema_model_cfg = model_cfg.get("ema_model")
    if ema_model_cfg is None or ema_model_cfg['target'] is None:
        ema_model = None
    else:
        ema_model_class, _ = import_class_from_string(ema_model_cfg['target'])
        ema_model = ema_model_class(main_model, **ema_model_cfg['params'])

    return main_model, class_name, ema_model


def initialize_dataloader(dataset_cfg):
    dataset_class, _ = import_class_from_string(dataset_cfg['target'])
    transform = transforms.Compose([
        transforms.Resize((dataset_cfg['image_size'], dataset_cfg['image_size'])),
        transforms.ToTensor()
    ])
    dataset = dataset_class(dataset_cfg['data_dir'], transform=transform)
    train_size = int(dataset_cfg['train_ratio'] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(dataset=train_dataset, batch_size=dataset_cfg['batch_size'], shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=dataset_cfg['batch_size'], shuffle=False)
    return train_loader, test_loader


def set_default_workpath(func, script_path):
    try:
        work_path = func(script_path)
    except Exception as e:
        print(f"执行函数时发生错误: {e}")
    try:
        os.chdir(work_path)
    except FileNotFoundError:
        print(f"指定的目录 {work_path} 不存在。")


def generate_unique_filepath(base_path, class_name, state='train'):
    number = 0
    while True:
        run_path = f"{class_name}_{state}_{number}".lower()
        run_path = os.path.join(base_path, run_path)
        if not os.path.exists(run_path):
            os.makedirs(run_path)
            return run_path
        number += 1
