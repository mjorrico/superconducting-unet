from scipy.spatial import Voronoi
from copy import deepcopy

import numpy as np
import cv2

import matplotlib.pyplot as plt

IMG_SIZE = 1000
TWOPI = 2 * np.pi

# def angle_gen()


def circle_multiplier(n: int, offset: float = 0):
    angles = np.linspace(
        offset,
        TWOPI * (1 - 1 / n) + offset,
        n,
        dtype=np.float32,
    )
    cos = np.cos(angles).reshape(-1, 1)
    sin = np.sin(angles).reshape(-1, 1)
    return np.concatenate([cos, sin], axis=1)


def generate_hex_centroids(
    hex_width: float,
    start_layer: int,
    end_layer: int,
    origin: np.ndarray,
    n_batch: int,
    variance: float,
):
    d = 2 * np.sin(np.pi / 3) * hex_width
    orig = origin.reshape(2)
    direction_vec = circle_multiplier(6, TWOPI / 3)
    centroids = []

    # centroid generation starts
    for layer_idx in range(start_layer, end_layer):
        # circle_multiplier(1) = [[1 0]]
        cell = orig + layer_idx * d * circle_multiplier(1)[0]
        for v in direction_vec:
            for _ in range(layer_idx):
                cell += d * v
                centroids.append(deepcopy(cell.tolist()))

    centroids = np.array(centroids)  # shape: (162, 2)
    n_centers = len(centroids)
    # centroid generation ends

    # shift centroid starts
    random_angles = np.random.uniform(0, TWOPI, (n_batch, n_centers))
    random_angles_sin = np.sin(random_angles)
    random_angles_cos = np.cos(random_angles)
    random_vecs = np.stack((random_angles_cos, random_angles_sin), axis=2)
    random_shift = np.random.uniform(0, d / 2 * variance, (n_batch, n_centers, 1))

    centroids_r = centroids.reshape(1, 162, 2) + random_vecs * random_shift
    # shift centroid ends

    # print(np.shape(random_vecs))
    # print(np.shape(random_shift))
    # print(np.shape(random_vecs * random_shift))

    return centroids.astype(np.int32), centroids_r.astype(np.int32)


def generate(
    n: int,
    rad: float,
    hex_width: float,
    n_batch: int = 200,
):
    origin = np.array([IMG_SIZE / 2] * 2, dtype=np.int32)
    coat_n = 360

    # coating polygon generation starts
    coat_components = circle_multiplier(coat_n).reshape(1, coat_n, 2)

    coat_base = np.ones((n_batch, coat_n, 1), np.float32)
    coat_inner_rad = coat_base * (rad + np.random.randint(0, 5, (n_batch, coat_n, 1)))
    coat_outer_rad = coat_inner_rad + np.random.randint(3, 6, (n_batch, coat_n, 1))

    coat_inner_poly = origin + coat_inner_rad * coat_components
    coat_outer_poly = origin + coat_outer_rad * coat_components

    coat_inner_poly = coat_inner_poly.astype(np.int32)
    coat_outer_poly = coat_outer_poly.astype(np.int32)
    # coating polygon generation ends

    # sub-elements polygons generation starts
    hex_poly, hex_poly_r = generate_hex_centroids(30, 2, 8, origin, n_batch, 0.5)
    # sub-elements polygons generation ends

    return coat_inner_poly, coat_outer_poly, hex_poly, hex_poly_r


coat_i, coat_o, hex_centroids, hex_centroids_r = generate(1000, 300, 50)
imgs = np.zeros((200, 1000, 1000))
print(np.shape(hex_centroids))
print(np.shape(hex_centroids_r))
print(np.shape(coat_i))
print(np.shape(coat_o))

for i in range(200):
    base = np.zeros((1000, 1000), np.uint8)
    # cv2.fillPoly(base1, [coat_i[i], coat_o[i]], 255)
    for c in hex_centroids:
        cv2.circle(base, c, 5, 140, 1)

    for c in hex_centroids_r[i]:
        cv2.circle(base, c, 5, 255, 1)
    cv2.imwrite(f"img_{i}.jpg", base)

    if i > 10:
        break

# base = np.zeros((1000, 1000), ui8)
# print(poly[0:1, :10, :].astype(ui16))
# tp = np.array([[[100, 100], [100, 200], [200, 100]]])
# print(np.shape(tp))
# cv2.fillPoly(base, tp, 255)
# cv2.imwrite("test.jpg", base)
