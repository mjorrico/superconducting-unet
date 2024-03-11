import torch
from model import WireUNet
from dataset import SuperWireDataset
from utils import save_predictions_as_imgs, load_checkpoint
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader


model = WireUNet(in_channels=1, out_channels=4, features=[64, 128, 256, 512])
load_checkpoint(torch.load("models/m2.pth.tar"), model)
model = model.to(device="cuda")

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

dataset = SuperWireDataset("data/val/img", "data/val/mask", val_transform, False)
loader = DataLoader(dataset, 8)

save_predictions_as_imgs(
    loader,
    model,
    folder="testexport/",
    is_binary=False,
)
