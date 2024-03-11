import os
import torch
import torchvision
from dataset import SuperWireDataset
from torch.utils.data import DataLoader


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("Saving checkpoint...")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("Loading checkpoint...")
    model.load_state_dict(checkpoint["state_dict"])
    print("Model loaded!")


def get_loaders(
    train_img_dir,
    train_mask_dir,
    val_img_dir,
    val_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
):
    train_ds = SuperWireDataset(
        image_dir=train_img_dir,
        mask_dir=train_mask_dir,
        transform=train_transform,
        is_train=True,
    )

    train_dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_ds = SuperWireDataset(
        image_dir=val_img_dir,
        mask_dir=val_mask_dir,
        transform=val_transform,
        is_train=False,
    )

    val_dataloader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_dataloader, val_dataloader


def probability2segment(probability_tensor, with_background=False):
    n_channels = 4 if with_background else 3
    preds = probability_tensor[:, :n_channels, :, :]
    preds_maxval, preds_maxidx = torch.max(preds, dim=1)
    preds = (preds == preds_maxval.unsqueeze(1)).int()
    return preds


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)[:, :3, :, :]
            preds = probability2segment(preds)
            num_correct = (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8)

    print(
        f"Accuracy: {num_correct / num_pixels}, dice score: {dice_score / len(loader)}"
    )


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", is_binary=True, device="cuda"
):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = model(x)
            if is_binary:
                preds = probability2segment(preds, with_background=True)
            else:
                preds = preds[:, :3, :, :]
            # preds_maxval, preds_maxidx = torch.max(preds, dim=1)
            # preds = (preds == preds_maxval.unsqueeze(1)).float() * 255

        preds = preds[:, :3, :, :].float()
        y = y[:, :3, :, :].float()

        pred_path = os.path.join(folder, f"pred_{idx}.png")
        truth_path = os.path.join(folder, f"truth_{idx}.png")
        torchvision.utils.save_image(preds, pred_path)
        torchvision.utils.save_image(y, truth_path)
