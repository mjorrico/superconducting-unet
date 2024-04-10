import torch
from .model import WireUNet

# from dataset import SuperWireDataset
from .utils import save_predictions_as_imgs, load_checkpoint

# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
from glob import glob
import sys
from tqdm import tqdm

IMG_FORMATS = ["png", "jpg", "jpeg", "tiff", "tif", "gif"]


def usage():
    return "Usage: python3 pred.py -source <image|folder> [-dest <folder>] [-animate <true|false>]"


def parse_args():
    dest = None
    animate = False
    if "-source" not in sys.argv:
        raise ValueError("Error: Path to image(s) must be provided.")

    if len(sys.argv) % 2 != 1:
        raise ValueError(f"{usage()}")

    flags = sys.argv[1::2]
    vals = sys.argv[2::2]

    for f, v in zip(flags, vals):
        if f == "-source":
            source = v
        elif f == "-dest":
            dest = v
        elif f == "-animate":
            animate = v
        else:
            raise ValueError(f'{usage()}\nError: Unrecognized flag "{f}".')

    animate = True if animate.lower() == "true" else False
    dest = "pred_images" if not dest else dest
    dest += "/" if dest[-1] != "/" else ""

    return source, dest, animate


def prepare_input(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = np.min(np.shape(img)[:2])
    size = np.int32(np.round(size / 160)) * 160
    size = np.max((size, 160))
    img = cv2.resize(img, (size, size))
    npatch = int(size / 160)

    border_type = cv2.BORDER_REFLECT
    img = cv2.copyMakeBorder(img, 20, 20, 20, 20, border_type)
    img = cv2.copyMakeBorder(img, 20, 20, 20, 20, border_type)

    patches = []
    for i in range(npatch):
        x1 = i * 160
        x2 = x1 + 240
        for j in range(npatch):
            y1 = j * 160
            y2 = y1 + 240
            patches.append(img[x1:x2, y1:y2])
    return np.stack(patches, axis=0) / 255


def predict_input(patches, model):

    patches = torch.from_numpy(patches).unsqueeze(1)
    patches = patches.float().to(device="cuda")

    with torch.no_grad():
        pred = model(patches)[:, :3, :, :].float().to(device="cpu")

    return (pred.detach().numpy() * 255).astype(np.uint8)


def stitch_images(preds):
    preds = [np.moveaxis(pat, 0, -1)[40:200, 40:200] for pat in preds]
    npatch = np.sqrt(len(preds)).astype(np.int32)
    size = npatch * 160
    output_img = np.zeros((size, size, 3))

    for i in range(npatch):
        x1 = i * 160
        x2 = x1 + 160
        for j in range(npatch):
            y1 = j * 160
            y2 = y1 + 160
            pred_patch = preds[i * npatch + j]
            pred_patch = np.flip(pred_patch, axis=-1)
            output_img[x1:x2, y1:y2, :] = pred_patch

    return output_img.astype(np.uint8)


def main():
    src, dst, anim = parse_args()
    if src.split(".")[-1] not in IMG_FORMATS:
        src = os.path.join(src, "*")

    input_images = [img for img in glob(src) if img.split(".")[-1] in IMG_FORMATS]
    n_imgs = len(input_images)

    if n_imgs == 0:
        raise ValueError("Error: No input images. Check input directory.")
    print(f"\nPredicting {n_imgs} image(s).")
    print(f'Saving to "{dst}".')
    print(f"With animation: {anim}.")

    model = WireUNet(in_channels=1, out_channels=4, features=[16, 32, 64, 128])
    load_checkpoint(torch.load("models/30032024-16-3.pth.tar"), model)
    model = model.to(device="cuda")
    model.eval()

    for input_img in input_images:
        img = cv2.imread(input_img)
        patches = prepare_input(img)

        preds = predict_input(patches, model)

        output_img = stitch_images(preds)
        cv2.imwrite("test_stitch.png", output_img)
        break


class   WireSegmentor:
    def __init__(self, model_path):
        self.model = WireUNet(in_channels=1, out_channels=4, features=[16, 32, 64, 128])
        load_checkpoint(torch.load(model_path), self.model)
        self.model = self.model.to(device="cuda")
        self.model.eval()

    def predict(self, path, destination, is_animate):
        if os.path.isdir(path):
            path = os.path.join(path, "*")

        image_paths = [img for img in glob(path) if img.split(".")[-1] in IMG_FORMATS]

        if len(image_paths) == 0:
            raise ValueError("Error: No input images. Check input directory.")
        else:
            print(f"\nPredicting {len(image_paths)} image(s).")
            print(f'Saving to "{destination}".')
            print(f"With animation: {is_animate}.")

        for input_img in tqdm(image_paths):
            img = cv2.imread(input_img)
            patches = prepare_input(img)
            preds = predict_input(patches, self.model)
            output_img = stitch_images(preds)

            filename = input_img.split("/")[-1].split(".")[:-1]
            filename = "".join(filename)
            full_dest = os.path.join(destination, filename)
            cv2.imwrite(f"{full_dest}.png", output_img)


if __name__ == "__main__":
    main()
