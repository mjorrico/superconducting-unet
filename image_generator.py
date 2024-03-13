from datetime import datetime
from copy import deepcopy
import poly_generator as pg
import pyarrow as pa
import pyarrow.parquet as pq
import imageio
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
        void_size = len(chosen_void)
        cv2.bitwise_and(
            chosen_void,
            copper_mask[xpos : xpos + void_size, ypos : ypos + void_size],
            chosen_void,
        )
        cv2.bitwise_or(
            chosen_void,
            void_mask[xpos : xpos + void_size, ypos : ypos + void_size],
            chosen_void,
        )

        void_mask[xpos : xpos + void_size, ypos : ypos + void_size] = chosen_void


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
    zmat = np.zeros((imagesize, imagesize), dtype=np.uint8)

    # MASKS GENERATION
    coat_in, coat_out, hexagon_in, hexagon_out = pg.generate(
        radius,
        elementsize,
        variance,
        imagesize,
    )
    coat_mask = cv2.fillPoly(deepcopy(zmat), coat_out + coat_in, 255)
    subelement_mask = cv2.fillPoly(deepcopy(zmat), hexagon_out + hexagon_in, 255)
    copper_mask = cv2.fillPoly(deepcopy(zmat), coat_in + hexagon_out + hexagon_in, 255)
    copper_inner_mask = cv2.fillPoly(deepcopy(zmat), hexagon_in, 255)
    void_mask = deepcopy(zmat)

    # ERROR CHECKING
    fillrate = np.sum(copper_mask > 0)
    if fillrate < 0.01:
        raise ValueError("Copper is too small. Increase wire radius.")
    elif fillrate < 0.05:
        warnings.warn("Copper fillrate is lower than 5%.", UserWarning)

    # GENERATING VOIDS
    s = np.sum(small_void_weights)
    N_small_voids = np.array(small_void_weights) / s * n_small_void
    N_small_voids = N_small_voids.astype(np.int32)

    for size, N_void in zip(small_void_sizes, N_small_voids):
        void_group = voiddict[size]
        spread_void(void_group, N_void, copper_mask, void_mask)
    spread_void(voiddict[20], n_big_void, copper_inner_mask, void_mask)

    # FINALIZING MASKS
    subelement_mask = np.subtract(subelement_mask, void_mask)
    copper_mask = np.subtract(copper_mask, void_mask)
    mask = np.stack((copper_mask, subelement_mask, void_mask), axis=-1)

    # COLORING
    finalimg = deepcopy(zmat)
    finalimg[coat_mask == 255] = 20
    finalimg[copper_mask == 255] = 135
    finalimg[subelement_mask == 255] = 200
    finalimg[void_mask == 255] = 50

    # ROTATION
    rotation_angle = np.random.uniform(0, 360)
    c = int(imagesize / 2)
    t_mat = cv2.getRotationMatrix2D([c, c], rotation_angle, 1)
    t_flag = cv2.INTER_NEAREST
    finalimg = cv2.warpAffine(finalimg, t_mat, np.shape(finalimg), flags=t_flag)
    mask = cv2.warpAffine(mask, t_mat, np.shape(finalimg), flags=t_flag)

    # BACKGROUND
    finalimg[finalimg == 0] = 63

    # BLUR
    cv2.GaussianBlur(finalimg, (5, 5), 50, finalimg, 50)

    # NOISE
    noise = np.random.normal(0, 0.05 * finalimg + 4, (imagesize, imagesize))
    finalimg = np.subtract(finalimg, noise).astype(np.uint8)

    return finalimg, mask


def save_wire_image(voiddict, img_dir, mask_dir, file_prefix, img_idx):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
    img, mask = generate_superconducting_wire(voiddict)
    img_name = os.path.join(img_dir, f"{file_prefix}-{img_idx}.jpg")
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

    img, mask = generate_superconducting_wire(voiddict)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    animation_name = os.path.join(folderpath, f"{datestr}.gif")
    imageio.mimsave(animation_name, [img, mask], fps=0.5, format="GIF", loop=0)


voiddict = read_void("void/")

# generate_superconducting_wire(voiddict)

start = time()
generate("test-normal/", voiddict, 5)
print(f"Time elapsed: {time() - start} seconds")
