from scipy.spatial import Voronoi, voronoi_plot_2d
from copy import deepcopy
from time import time, sleep

import numpy as np
import cv2

import matplotlib.pyplot as plt

TWOPI = 2 * np.pi


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


def generate_hex_centroids(hex_width: float, origin: np.ndarray, variance: float):
    d = 2 * np.sin(np.pi / 3) * hex_width
    direction_vec = circle_multiplier(6, TWOPI / 3)
    centroids = []

    # centroid generation starts
    for layer_idx in range(2, 8):
        cell = origin + layer_idx * d * np.array([1, 0])
        for v in direction_vec:
            for _ in range(layer_idx):
                cell += d * v
                centroids.append(deepcopy(cell.tolist()))

    centroids = np.array(centroids)  # shape: (162, 2)
    N = len(centroids)
    # centroid generation ends

    # shift centroid starts
    random_angles = np.random.uniform(0, TWOPI, N)
    random_angles_sin = np.sin(random_angles)
    random_angles_cos = np.cos(random_angles)
    random_vecs = np.stack((random_angles_cos, random_angles_sin), axis=-1)
    random_shift = np.random.uniform(0, d / 2 * variance, (N, 1))
    centroids = centroids + random_vecs * random_shift

    v = Voronoi(centroids)
    centroids = centroids[12:120]
    vertices = [[v.vertices[j] for j in v.regions[i]] for i in v.point_region[12:120]]
    vertices = [np.array(v) for v in vertices]

    outer = []
    c = 0
    for i, vertex in enumerate(vertices):
        shift_coeff = np.random.uniform(0.75, 0.86, (len(vertex), 1))
        outer_vertex = (vertex - centroids[i]) * shift_coeff + centroids[i]
        outer.append(outer_vertex)

    inner = []
    for i, vertex in enumerate(outer):
        shift_coeff = np.random.uniform(0.6, 0.7, (len(vertex), 1))
        inner_vertex = (vertex - centroids[i]) * shift_coeff + centroids[i]
        inner.append(inner_vertex)

    outer = [v.astype(np.int32) for v in outer]
    inner = [v.astype(np.int32) for v in inner]

    return outer, inner, centroids


def generate_coating(rad: float, origin: np.ndarray):
    N = 360
    coat_components = circle_multiplier(N)

    coat_base = np.ones((N, 1), np.float32)
    coat_inner_rad = coat_base * (rad + np.random.randint(0, 5, (N, 1)))
    coat_outer_rad = coat_inner_rad + np.random.randint(3, 6, (N, 1))

    coat_inner_poly = origin + coat_inner_rad * coat_components
    coat_outer_poly = origin + coat_outer_rad * coat_components

    coat_inner_poly = coat_inner_poly.astype(np.int32)
    coat_outer_poly = coat_outer_poly.astype(np.int32)

    return coat_inner_poly, coat_outer_poly


def generate(rad: float, hex_width: float, variance: float = 0.1, imgsize: int = 1000):
    origin = np.array([imgsize / 2] * 2, dtype=np.int32)

    # coating polygon generation
    coating_inner, coating_outer = generate_coating(rad, origin)

    # sub-elements polygons generation
    outer, inner, c = generate_hex_centroids(hex_width, origin, variance)

    return [coating_inner], [coating_outer], inner, outer, c
