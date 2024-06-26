import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from .model import WireUNet
import numpy as np

from .utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    probability2segment,
)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CUDA_VERSION = torch.version.cuda if DEVICE == "cuda" else "-"
BATCH_SIZE = 5
NUM_EPOCHS = 40000
NUM_WORKERS = 4
IMAGE_HEIGHT = 240
IMAGE_WIDTH = 240
PIN_MEMORY = True
LOAD_MODEL = True
TRAIN_IMG_DIR = "data/transfer_learning/img"
TRAIN_MASK_DIR = "data/transfer_learning/mask"
VAL_IMG_DIR = "data/val/img"
VAL_MASK_DIR = "data/val/mask"
FEATURES = [16, 32, 64, 128]
TRANSFER_LEARNING = True

print(f"Device: {DEVICE}. CUDA version: {CUDA_VERSION}.")


def train_fn(loader, model, optimizer, loss_fn, scaler):
    # loop = tqdm(loader)

    # for batch_idx, (data, targets) in enumerate(loop):
    for batch_idx, (data, targets) in enumerate(loader):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        # forward pass
        with torch.cuda.amp.autocast():
            prediction = model(data)
            loss = loss_fn(prediction, targets)

        # backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm info
        # loop.set_postfix(loss=loss.item())

        # if batch_idx == 100:
        #     break


def train():
    train_transform = A.Compose(
        [
            A.RandomGridShuffle(grid=(3, 3), p=1),
            A.RandomCrop(IMAGE_HEIGHT, IMAGE_WIDTH, True),
            A.RandomBrightnessContrast((-0.2, 0.2), (-0.1, 0.1), p=1),
            A.Sharpen(alpha=(0.2, 0.5), p=0.1),
            A.PixelDropout(p=0.8),
            A.Solarize(threshold=np.random.randint(0, 255), p=0.3),
            A.InvertImg(p=0.3),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.MultiplicativeNoise(
                (0.8, 1.2),
                elementwise=True,
                per_channel=False,
                p=0.3,
            ),
            A.Normalize(
                mean=0,
                std=1,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    val_transform = A.Compose(
        [
            A.RandomCrop(IMAGE_HEIGHT, IMAGE_WIDTH, True),
            A.Normalize(
                mean=0,
                std=1,
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    model = WireUNet(in_channels=1, out_channels=4, features=FEATURES)
    if LOAD_MODEL:
        load_checkpoint(torch.load("models/BESTMODEL-5APR.pth.tar"), model)
        if TRANSFER_LEARNING:
            for name, param in model.named_parameters():
                name_split = name.split(".")
                if name_split[0] in ["output", "expansive"]:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    model = model.to(device=DEVICE)

    loss_weights = torch.Tensor([1.1, 0.9, 0.6, 0.4]).to(device=DEVICE)
    loss_fn = nn.CrossEntropyLoss(loss_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    scaler = torch.cuda.amp.GradScaler()
    for epoch in tqdm(range(NUM_EPOCHS)):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        if epoch % 1000 == 999:
            save_checkpoint(checkpoint, filename=f"08042024-{epoch}.pth.tar")

        # check accuracy
        # check_accuracy(val_loader, model, device=DEVICE)

        # save samples
        # save_predictions_as_imgs(
        #     val_loader,
        #     model,
        #     folder="saved_images/",
        #     is_binary=False,
        # )


if __name__ == "__main__":
    train()
