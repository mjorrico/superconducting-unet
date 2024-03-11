import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class SuperWireDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform, is_train=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(self.image_dir)
        self.is_train = is_train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        filename = self.images[index]
        image_path = os.path.join(self.image_dir, filename)
        mask_path = os.path.join(self.mask_dir, filename.replace(".jpg", ".png"))

        image = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)
        mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.uint8)

        aug = self.transform(image=image, mask=mask)
        image = aug["image"]
        mask = aug["mask"]

        if self.is_train:
            mask_background = (torch.sum(mask, dim=-1) == 0).unsqueeze(-1) * 255
            mask = torch.concatenate((mask, mask_background), dim=-1)
        mask = torch.moveaxis(mask, -1, 0)
        mask[mask == 255] = 1 # mask isn't normalized by albumentations.Normalize

        return image, mask
