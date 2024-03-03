import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class SuperWireDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform, augment_factor):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augment_factor = augment_factor
        self.images = os.listdir(self.image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])

        image = np.array(Image.open(image_path).convert("L"), dtype=np.uint8)
        mask = np.array(Image.open(mask_path).convert("RGB"), dtype=np.float32)
        mask[mask == 255] = 1.0

        image = self.transform(image=image, mask=mask)
        image = aug["image"]
        mask = aug["mask"]

        return image, mask

from time import time
if __name__ == "__main__":
    data = SuperWireDataset("data/train/img", "data/train/mask")
    start = time()
    for i in range(1000):
        image, mask = data[i+10000]
    print(f"Time to fetch: {time() - start} seconds")
    print(np.shape(image), np.shape(mask))