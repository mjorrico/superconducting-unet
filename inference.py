import torch
from model import WireUNet
from dataset import SuperWireDataset
from utils import save_predictions_as_imgs, load_checkpoint
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader


model = WireUNet(in_channels=1, out_channels=4, features=[16, 32, 64, 128])
load_checkpoint(torch.load("models/170324-16-2.pth.tar"), model)
model = model.to(device="cuda")
model.eval()

val_transform = A.Compose(
    [
        A.RandomCrop(240, 240, True),
        A.Normalize(
            mean=0,
            std=1,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

# dataset = SuperWireDataset("data/val/img", "data/val/mask", val_transform, False)
# loader = DataLoader(dataset, 8)

# save_predictions_as_imgs(
#     loader,
#     model,
#     folder="testexport/",
#     is_binary=False,
# )

import numpy as np
from PIL import Image
import os
import torchvision
import cv2

read_dir = "data/val/img/real1.jpg"
write_dir = "testdir"

image = cv2.imread(read_dir)
print(np.shape(image))
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, (1000, 1000))
print(f"Read (Min: {np.min(image)}, Max: {np.max(image)}, Shape: {np.shape(image)})")

aug = val_transform(image=image)
image = aug["image"].unsqueeze(0).to(device="cuda")

host_input = image.to(device="cpu").detach().numpy() * 255
host_input = host_input.astype(np.uint8)[0]
host_input = np.moveaxis(host_input, 0, -1)
print(f"Processed (Min: {np.min(host_input)}, Max: {np.max(host_input)}, Shape: {np.shape(host_input)})")
print(f"Processed tensor (Min: {torch.min(image)}, Max: {torch.max(image)}, Shape: {image.shape})")
cv2.imwrite("testoutput/cropped.png", host_input)

with torch.no_grad():
    pred = model(image)[:, :3, :, :].float()
print(f"Predict tensor (Min: {torch.min(pred)}, Max: {torch.max(pred)}, Shape: {pred.shape})")

folder = "testoutput"
os.makedirs(folder, exist_ok=True)
pred_path = os.path.join(folder, f"pred_real.png")
torchvision.utils.save_image(pred, pred_path)