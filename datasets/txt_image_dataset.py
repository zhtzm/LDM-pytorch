import json
import os

from PIL import Image
from torch.utils.data import Dataset


class TxtImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.info_file = os.path.join(self.data_dir, 'info.json')
        self.transform = transform

        with open(self.info_file, 'r') as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = os.path.join(self.data_dir, 'images', item['image_path'])
        tags = item['tags']

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found.")

        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, tags
