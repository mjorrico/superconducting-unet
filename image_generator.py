from datetime import datetime
from copy import deepcopy
import poly_generator as pg
import pickle
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
    filepath = os.path.join(path, "void*.pickle")
    files = glob.glob(filepath)

    if len(files) == 0:
        raise RuntimeError

    outputdict = {}
    for file in files:
        with open(file, "rb") as f:
            arr = pickle.load(f)
            size = np.shape(arr)[1]
            outputdict[size] = arr

    return outputdict


def generate_superconducting_wire(voiddict: dict):
    radius = np.random.uniform(400, 450)
    elementsize = (radius - 400) * 6 / 50 + 27
    elementsize = np.random.uniform(elementsize, elementsize + 6)
    variance = 0.4
    imagesize = 1000
    n_big_void = 200
    n_small_void = 300
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
        raise ValueError("Cable is too small. Increase wire radius.")
    elif fillrate < 0.05:
        warnings.warn("Cable fillrate is lower than 5%.", UserWarning)

    # GENERATING VOIDS
    s = np.sum(small_void_weights)
    N_small_voids = np.array(small_void_weights) / s * n_small_void
    N_small_voids = N_small_voids.astype(np.int32)

    for size, N_void in zip(small_void_sizes, N_small_voids):
        void_group = voiddict[size]
        spread_void(void_group, N_void, copper_mask, void_mask)
    spread_void(voiddict[20], n_big_void, copper_inner_mask, void_mask)

    # PREPARING MASKS
    rotation_angle = np.random.uniform(0, 360)
    c = int(imagesize / 2)
    t_mat = cv2.getRotationMatrix2D([c, c], rotation_angle, 1)
    t_flag = cv2.INTER_NEAREST

    cv2.warpAffine(coat_mask, t_mat, np.shape(zmat), coat_mask, t_flag)
    cv2.warpAffine(subelement_mask, t_mat, np.shape(zmat), subelement_mask, t_flag)
    cv2.warpAffine(copper_mask, t_mat, np.shape(zmat), copper_mask, t_flag)
    cv2.warpAffine(copper_inner_mask, t_mat, np.shape(zmat), copper_inner_mask, t_flag)
    cv2.warpAffine(void_mask, t_mat, np.shape(zmat), void_mask, t_flag)

    # COLORING
    finalimg = deepcopy(zmat)
    finalimg[copper_mask == 255] = 150
    finalimg[subelement_mask == 255] = 200

    # surface = np.random.uniform(size=(imagesize, imagesize))
    # surfaceblur = 10
    # cv2.GaussianBlur(surface, (0, 0), surfaceblur, surface, surfaceblur)
    # minval = np.min(surface)
    # maxval = np.max(surface)
    # rangeval = maxval - minval
    # surface = (surface - minval) / rangeval * 2 - 1
    # surface *= 30
    # surface = (150 - surface).astype(np.uint8)
    # cv2.subtract(finalimg, copper_mask, finalimg)
    # surface = cv2.bitwise_and(surface, surface, mask=copper_mask)
    # cv2.add(finalimg, surface, finalimg)

    kerneld = np.ones((4, 4), np.uint8)
    pseudocopper_mask = cv2.dilate(void_mask, kerneld, cv2.BORDER_REFLECT)
    pseudocopper_mask = np.subtract(pseudocopper_mask, void_mask)
    pseudocopper_mask = cv2.bitwise_and(pseudocopper_mask, copper_inner_mask)
    finalimg[pseudocopper_mask == 255] = 200

    cv2.GaussianBlur(finalimg, (9, 9), 500, finalimg, 50)

    finalimg[void_mask == 255] = 70
    kernel1 = np.ones((3, 3), np.uint8)
    void_mask_in1 = cv2.erode(void_mask, kernel1, cv2.BORDER_REFLECT)
    finalimg[void_mask_in1 == 255] = 80
    kernel2 = np.ones((6, 6), np.uint8)
    void_mask_in2 = cv2.erode(void_mask, kernel2, cv2.BORDER_REFLECT)
    finalimg[void_mask_in2 == 255] = 90

    finalimg[coat_mask == 255] = 50
    finalimg[finalimg == 0] = 63

    cv2.GaussianBlur(finalimg, (3, 3), sigmaX=50, dst=finalimg)

    # # NOISE
    noise = np.random.normal(0, 0.04 * finalimg + 1, (imagesize, imagesize))
    finalimg = np.subtract(finalimg, noise)
    finalimg = np.clip(finalimg, 0, 255).astype(np.uint8)

    # FINALIZING MASKS
    subelement_mask[void_mask > 0] = 0
    copper_mask[void_mask > 0] = 0
    void_mask[void_mask > 0] = 255
    mask = np.stack((copper_mask, subelement_mask, void_mask), axis=-1)

    return finalimg, mask


def save_wire_image(voiddict, img_dir, mask_dir, file_prefix, img_idx):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder="little"))
    img, mask = generate_superconducting_wire(voiddict)
    img_name = os.path.join(img_dir, f"{file_prefix}-{img_idx}.jpg")
    mask_name = os.path.join(mask_dir, f"{file_prefix}-{img_idx}.png")
    status1 = cv2.imwrite(img_name, img)
    status2 = cv2.imwrite(mask_name, mask)
    if img_idx % 1000 == 0:
        print(f"Image {img_idx} saved!")

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
        # concurrent.futures.wait(futures)

    img, mask = generate_superconducting_wire(voiddict)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    animation_name = os.path.join(folderpath, f"{datestr}.gif")
    imageio.mimsave(animation_name, [img, mask], fps=0.5, format="GIF", loop=0)


if __name__ == "__main__":
    voiddict = read_void("void/")
    # generate_superconducting_wire(voiddict)
    start = time()
    generate("test-normal/", voiddict, 100)
    print(f"Time elapsed: {time() - start} seconds")
