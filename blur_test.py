import glob
import os
import numpy as np
import pickle
import cv2
import albumentations as A
from PIL import Image

# IMAGE_HEIGHT = 240
# IMAGE_WIDTH = 240

# train_transform = A.Compose(
#     [
#         A.RandomGridShuffle(grid=(3, 3),p=1),
#         A.RandomCrop(IMAGE_HEIGHT, IMAGE_WIDTH, True),
#         A.RandomBrightnessContrast((-0.2, 0.2), (-0.1, 0.1), p=1),
#         A.Sharpen(alpha=(0.2, 0.5), p=0.1),
#         A.PixelDropout(p=0.8),
#         A.Solarize(threshold=np.random.randint(0, 255), p=.3),
#         A.InvertImg(p=.3),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.MultiplicativeNoise(
#             (.8, 1.2),
#             elementwise=True,
#             per_channel=False,
#             p=0.3,
#         ),
#         # A.Normalize(
#         #     mean=0,
#         #     std=1,
#         #     max_pixel_value=255.0,
#         # ),
#         # ToTensorV2(),
#     ]
# )

# img = cv2.imread("output/img/4bd16d0759-00000.jpg")
# # img = cv2.imread("real1000.png")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# aug = train_transform(image=img)
# augimg = aug["image"]
# cv2.imwrite("augimg.png", augimg)

# img = cv2.imread("data/val/img/4bd16d0759-00000.jpg")
# print(np.shape(img))

# size = np.shape(img)[0]
# size = np.int32(np.round(size / 160)) * 160
# size = np.max((size, 160))
# print(size)
# img = cv2.resize(img, (size, size))

# border_type = cv2.BORDER_REFLECT
# img = cv2.copyMakeBorder(img, 20, 20, 20, 20, border_type)
# img = cv2.copyMakeBorder(img, 20, 20, 20, 20, border_type)
# cv2.imwrite("testborder1.png", img)

image_path = "data/val/img/real1.jpg"
img = cv2.imread(image_path)
print(np.shape(img))
img = cv2.resize(img, (960, 960))
print(np.shape(img))
cv2.imwrite("960real.png", img)