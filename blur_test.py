import pyarrow as pa
import pyarrow.parquet as pq
import glob
import os
import numpy as np
import pickle
import cv2

with open("surface/surfaces.pickle", "rb") as f:
    test = pickle.load(f)

cv2.imwrite("test.png", test[19])