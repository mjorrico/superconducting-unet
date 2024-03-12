from datetime import datetime
from copy import deepcopy
import poly_generator as pg
import pyarrow as pa
import pyarrow.parquet as pq
import glob
import os
from time import time

import concurrent.futures
import warnings
import numpy as np
import cv2


def spread_void(void_group, n_void, copper_mask, void_mask, n_trial=300):
    imagesize = len(copper_mask)
    for _ in range(n_void):
        is_succeed = False
        for _ in range(n_trial):  # multiple attempts to insert void
            xpos, ypos = np.random.randint(0, imagesize - 20, 2)
            xcheck = xpos + 10
            ycheck = ypos + 10

            if copper_mask[xcheck, ycheck] > 0:
                is_succeed = True
                break

        if not is_succeed:
            break

        chosen_void = deepcopy(void_group[np.random.randint(0, len(void_group))])
        cv2.bitwise_and(
            chosen_void,
            copper_mask[xpos : xpos + 20, ypos : ypos + 20],
            chosen_void,
        )
        cv2.bitwise_or(
            chosen_void,
            void_mask[xpos : xpos + 20, ypos : ypos + 20],
            chosen_void,
        )

        void_mask[xpos : xpos + 20, ypos : ypos + 20] = chosen_void


def read_void(path: str):
    void = {}
    for file in glob.glob(os.path.join(path, "void*.parquet")):
        size = int((file.split("-")[-1]).split(".")[0])
        vraw = pq.read_table(file)
        void[size] = np.stack(
            [np.array(vraw[file]).reshape(size, size) for file in vraw.column_names],
            axis=0,
        )
    return void


def generate_superconducting_wire(voiddict: dict):
    radius = 400
    elementsize = 31
    variance = 0.4
    imagesize = 1000
    n_big_void = 70
    n_small_void = 150
    small_void_weights = [1, 2, 2]
    small_void_sizes = [15, 10, 7]

    zerosmatrix = np.zeros((imagesize, imagesize), dtype=np.uint8)
    coat_in, coat_out, hexagon_in, hexagon_out = pg.generate(
        radius,
        elementsize,
        variance,
        imagesize,
    )

    copper_mask_outer = cv2.fillPoly(deepcopy(zerosmatrix), coat_in + hexagon_out, 255)

    fillrate = np.sum(copper_mask_outer > 0)
    if fillrate < 0.01:
        raise ValueError("Copper is too small. Increase wire radius.")
    elif fillrate < 0.05:
        warnings.warn("Copper fillrate is lower than 5%.", UserWarning)

    # BEGIN GENERATING SMALL VOID
    s = np.sum(small_void_weights)
    N_small_voids = np.array(small_void_weights) / s * n_small_void
    N_small_voids = N_small_voids.astype(np.int32)

    void_small_mask = np.zeros_like(copper_mask_outer)

    for size, N_void in zip(small_void_sizes, N_small_voids):
        for _ in range(N_void):

            is_succeed = False
            for _ in range(100):  # tries 100 times to place void on copper
                xpos, ypos = np.random.randint(0, imagesize - size, 2)
                xcheck = xpos + int(size / 2)
                ycheck = ypos + int(size / 2)

                if copper_mask_outer[xcheck, ycheck] > 0:
                    is_succeed = True
                    break

            if not is_succeed:
                break

            void_group = voiddict[size]
            chosen_void = deepcopy(void_group[np.random.randint(0, len(void_group))])
            cv2.bitwise_and(
                chosen_void,
                copper_mask_outer[xpos : xpos + size, ypos : ypos + size],
                chosen_void,
            )
            cv2.bitwise_or(
                chosen_void,
                void_small_mask[xpos : xpos + size, ypos : ypos + size],
                chosen_void,
            )

            void_small_mask[xpos : xpos + size, ypos : ypos + size] = chosen_void

    # BEGIN GENERATING BIG VOID
    copper_mask_inner = cv2.fillPoly(deepcopy(zerosmatrix), hexagon_in, 255)
    void_big_mask = np.zeros_like(copper_mask_inner)

    for _ in range(n_big_void):

        is_succeed = False
        for _ in range(300):  # tries 300 times to place void on copper
            xpos, ypos = np.random.randint(0, imagesize - 20, 2)
            xcheck = xpos + 10
            ycheck = ypos + 10

            if copper_mask_inner[xcheck, ycheck] > 0:
                is_succeed = True
                break

        if not is_succeed:
            break

        void_group = voiddict[20]
        chosen_void = deepcopy(void_group[np.random.randint(0, len(void_group))])
        cv2.bitwise_and(
            chosen_void,
            copper_mask_inner[xpos : xpos + 20, ypos : ypos + 20],
            chosen_void,
        )
        cv2.bitwise_or(
            chosen_void,
            void_big_mask[xpos : xpos + 20, ypos : ypos + 20],
            chosen_void,
        )

        void_big_mask[xpos : xpos + 20, ypos : ypos + 20] = chosen_void

    # BEGIN MASKS GENERATION
    coat_mask = cv2.fillPoly(deepcopy(zerosmatrix), coat_out + coat_in, 255)
    subelement_mask = cv2.fillPoly(deepcopy(zerosmatrix), hexagon_out + hexagon_in, 255)
    void_mask = np.add(void_small_mask, void_big_mask)
    copper_mask = cv2.add(copper_mask_outer, copper_mask_inner)
    copper_mask = cv2.subtract(copper_mask, void_mask)
    mask = np.stack((copper_mask, subelement_mask, void_mask), axis=-1)

    # BEGIN DRAWING
    finalimg = deepcopy(zerosmatrix)
    finalimg[coat_mask == 255] = 20
    finalimg[copper_mask == 255] = 135
    finalimg[subelement_mask == 255] = 200
    finalimg[void_mask == 255] = 50

    # BEGIN TRANSLATION
    rotation_angle = np.random.uniform(0, 360)
    c = int(imagesize / 2)
    t_mat = cv2.getRotationMatrix2D([c, c], rotation_angle, 1)
    t_flag = cv2.INTER_NEAREST
    finalimg = cv2.warpAffine(finalimg, t_mat, np.shape(finalimg), flags=t_flag)
    mask = cv2.warpAffine(mask, t_mat, np.shape(finalimg), flags=t_flag)

    # BACKGROUND
    finalimg[finalimg == 0] = 63

    return finalimg, mask


def save_wire_image(voiddict, img_dir, mask_dir, file_prefix, img_idx):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
    img, mask = generate_superconducting_wire(voiddict)
    img_name = os.path.join(img_dir, f"{file_prefix}-{img_idx}.png")
    mask_name = os.path.join(mask_dir, f"{file_prefix}-{img_idx}.png")
    status1 = cv2.imwrite(img_name, img)
    status2 = cv2.imwrite(mask_name, mask)


def generate(folderpath: str, voiddict: dict, N_img: int = 1):
    current_datetime = datetime.now()
    datestr = current_datetime.strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]
    img_dir = os.path.join(folderpath, "img")
    mask_dir = os.path.join(folderpath, "mask")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(save_wire_image, voiddict, img_dir, mask_dir, datestr, i)
            for i in range(N_img)
        ]
        concurrent.futures.wait(futures)


voiddict = read_void("void/")

# generate_superconducting_wire(voiddict)

start = time()
generate("test-normal/", voiddict, 8)
end = time() - start
print(f"Time elapsed: {end} seconds")
