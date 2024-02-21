from scipy.spatial import Voronoi
from copy import deepcopy

import numpy as np
import cv2

IMG_SIZE = 1000


def angle_multiplier(angles: list[float]):
    cos = np.cos(angles).reshape(-1, 1)
    sin = np.sin(angles).reshape(-1, 1)
    return np.concatenate([cos, sin], axis=1)


def generate(n: int, n_batch: int = 200):
    origin = np.array([IMG_SIZE / 2] * 2, dtype=np.int32).reshape(1, 1, 2)

    coating_n_poly = 360
    coating_angles = np.linspace(0.01, 2*np.pi, coating_n_poly)
    coating_angle_multiplier = angle_multiplier(coating_angles)
    coating_outer_radius = np.random.uniform(n_batch, coating_n_poly, 2)
    return np.random.uniform()