from scipy.spatial import Voronoi
from copy import deepcopy

import numpy as np
import cv2


def angle_multiplier(angles: list[float]):
    cos = np.cos(angles).reshape(-1, 1)
    sin = np.sin(angles).reshape(-1, 1)
    return np.concatenate([cos, sin], axis=1)


def generate_background(n: int):
    img = np.random.normal(60, 20, (n, n))
    img = img * (img > 0)
    img = img.astype(np.uint8)
    img[-1, 0] = 0
    img[-1, -1] = 255
    return img


def generate_copper(radius: int, center: list):
    angles = np.linspace(0, np.pi * 2 * 359 / 360, 360)
    inner_rad = np.random.randint(0, 5, 360) + radius
    outer_rad = np.random.randint(4, 6, 360) + inner_rad

    inner_points = inner_rad[:, np.newaxis] * angle_multiplier(angles)
    inner_points += np.array(center)
    outer_points = outer_rad[:, np.newaxis] * angle_multiplier(angles)
    outer_points += np.array(center)

    return [outer_points.astype(np.int32), inner_points.astype(np.int32)]


def initialize_centroids(cell_edge: int, variance: float):
    origin = np.array([500, 500])
    r = 2 * np.sin(np.pi / 3) * cell_edge
    start_layer = 2
    end_layer = 8

    angles = np.linspace(2 / 3 * np.pi, 7 / 3 * np.pi, 6)
    vector = angle_multiplier(angles)

    centers = []

    for L in range(start_layer, end_layer):
        cell = origin + L * r * angle_multiplier(0)[0, :]
        for v in vector:
            for _ in range(L):
                cell += r * v
                centers.append(deepcopy(cell.tolist()))

    centers = np.array(centers)
    n = len(centers)
    random_angles = angle_multiplier(np.random.uniform(0, 2 * np.pi, (n, 1)))
    random_lengths = np.random.uniform(0, variance, (n, 1))
    centers += random_lengths * random_angles

    v = Voronoi(centers)
    centers = v.points[12:120]  # excluding inner and outer-most layer 108 x 2
    cent_region = [
        [v.vertices[j] for j in v.regions[i]] for i in v.point_region[12:120]
    ]  # 108 x 6 x 2
    cent_region = np.array(cent_region, dtype=np.int32)

    # Random shift
    shift_coeff = np.random.uniform(0.8, 0.9, (108, 6, 1))
    outer_region = (cent_region - centers.reshape(108, 1, 2)) * shift_coeff
    outer_region += centers.reshape(108, 1, 2)

    shift_coeff = np.random.uniform(0.6, 0.7, (108, 6, 1))
    inner_region = (outer_region - centers.reshape(108, 1, 2)) * shift_coeff
    inner_region += centers.reshape(108, 1, 2)

    return [centers, outer_region.astype(np.int32), inner_region.astype(np.int32)]


def apply_mask(image, mask, noise_func):
    shape = np.shape(image)[:2]
    noise = noise_func(shape).astype(np.uint8)
    noise = cv2.bitwise_and(noise, noise, mask=mask)
    final_image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    final_image = cv2.add(final_image, noise)
    return final_image


def main():
    size = 1000
    blank = np.zeros((size, size), dtype=np.uint8)

    outer_copper, inner_copper = generate_copper(450, (500, 500))
    centers, outer_sub_element, inner_sub_element = initialize_centroids(35, 10)

    mask_copper_in = cv2.fillPoly(blank.copy(), [inner_copper], 255)
    mask_copper_out = cv2.fillPoly(blank.copy(), [outer_copper], 255)
    mask_sub_in = cv2.fillPoly(blank.copy(), inner_sub_element, 255)
    mask_sub_out = cv2.fillPoly(blank.copy(), outer_sub_element, 255)

    mask_subelement = cv2.subtract(mask_sub_out, mask_sub_in)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # mask_subelement = cv2.morphologyEx(mask_subelement, cv2.MORPH_CLOSE, kernel)
    mask_subelement = cv2.morphologyEx(mask_subelement, cv2.MORPH_OPEN, kernel)
    # mask_subelement = cv2.dilate(mask_subelement, kernel)
    # mask_subelement = cv2.erode(mask_subelement, kernel, iterations=2)
    # mask_subelement = cv2.dilate(mask_subelement, kernel)

    mask_copper_out = cv2.subtract(mask_copper_out, mask_copper_in)
    mask_copper_in = cv2.subtract(mask_copper_in, mask_subelement)

    # cv2.imshow("mask_copperin", mask_copper_in)
    # cv2.imshow("mask_copperout", mask_copper_out)
    # cv2.imshow("mask_subelement", mask_subelement)

    gray_image = generate_background(size)

    f = lambda s: np.random.randint(10, 40, s)
    gray_image = apply_mask(gray_image, mask_copper_out, f)

    f = lambda s: np.random.randint(100, 140, s)
    gray_image = apply_mask(gray_image, mask_copper_in, f)

    f = lambda s: np.random.randint(200, 230, s)
    gray_image = apply_mask(gray_image, mask_subelement, f)

    blurred_image = cv2.GaussianBlur(
        gray_image, (5, 5), sigmaX=10, sigmaY=10, borderType=cv2.BORDER_DEFAULT
    )
    
    blurred_image = cv2.add(blurred_image.astype(np.int16), np.random.randint(-30, 30, (size, size), dtype=np.int16)).astype(np.uint8)

    color_image = np.zeros((size, size, 3), dtype=np.uint8)
    color_image[:, :, 0] = mask_copper_in
    color_image[:, :, 1] = mask_subelement

    cv2.imshow("color_image", color_image)
    cv2.imshow("gray_image", blurred_image)

    gray_colorized = cv2.cvtColor(blurred_image, cv2.COLOR_GRAY2RGB)
    overlay_image = cv2.addWeighted(gray_colorized, 0.5, color_image, 0.05, 0)
    cv2.imshow("overlay_image", overlay_image)

    cv2.imwrite("generated_gray.png", blurred_image)
    cv2.imwrite("generated_mask.png", color_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
