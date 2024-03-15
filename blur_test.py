import pyarrow as pa
import pyarrow.parquet as pq
import glob
import os
import numpy as np
import pickle
import cv2

def getrange(arr):
    return np.min(arr), np.max(arr)


surfaces = []
for _ in range(500):
    img = np.random.uniform(0, 255, (1000, 1000))
    blur = np.random.uniform(20, 50)
    cv2.GaussianBlur(img, (0, 0), blur, img, blur)
    minval, maxval = getrange(img)
    img = (img - minval) / (maxval - minval) * 255
    img = img.astype(np.uint8)
    cv2.imwrite("test.png", img)
    tast = img / 6 + (135 - 127 / 6)
    cv2.imwrite("tast.png", tast)
    print(blur, np.mean(img), np.mean(tast), getrange(tast))
    break