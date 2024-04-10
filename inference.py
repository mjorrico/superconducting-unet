import numpy as np
import cv2
from glob import glob
import os
import albumentations as A


folder = "data/transfer_learning"
photopath = "1312-4"
img_path = os.path.join("img", photopath + ".jpg")
mask_path = os.path.join("msk", photopath + ".png")

img_raw = cv2.imread(img_path)
msk = cv2.imread(mask_path)

img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
# msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)

img = cv2.resize(img, (1000, 1000))
msk = cv2.resize(msk, (1000, 1000))
img_raw = cv2.resize(img_raw, (1000, 1000))

# 1. SET TRANSFORMATION ARGUMENTS FIRST
# 2. SET PROBABILITY LATER

IMAGE_HEIGHT, IMAGE_WIDTH = 240, 240

train_transform = A.Compose(
    [
        A.Rotate(p=1),
        A.RandomCrop(IMAGE_HEIGHT, IMAGE_WIDTH, True),
        A.RandomBrightnessContrast((0.3, 0.4), (-0.8, -0.4), p=1),
        A.PixelDropout(dropout_prob=0.04, p=0.9),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.MultiplicativeNoise(
            (0.8, 1.2),
            elementwise=True,
            per_channel=False,
            p=0.3,
        ),
    ]
)

dst = "transformation_test"

for i in range(20):
    aug = train_transform(image=img, mask=msk)
    img_aug = aug["image"]
    msk_aug = aug["mask"]
    img_aug = cv2.cvtColor(img_aug, cv2.COLOR_GRAY2BGR)
    result = np.zeros((240, 240*3, 3))
    result[0:240, 0:240, :] = img_aug
    result[0:240, 240:240*2, :] = img_raw
    result[0:240, 240*2:240*3, :] = msk_aug
    cv2.imwrite(os.path.join(dst, f"{i}.jpg"), result)