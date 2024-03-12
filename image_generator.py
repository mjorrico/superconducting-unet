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


def apply_mask(image, mask, mean, std):
    shape = np.shape(image)[:2]
    noise = np.random.randn(*shape) * std + mean
    noise = np.clip(noise, 1, 254)
    noise = noise.astype(np.uint8)
    noise = cv2.bitwise_and(noise, noise, mask=mask)
    final_image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    final_image = cv2.add(final_image, noise)
    return final_image


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


def chain_affine_transformation(M0, M1):
    T0 = np.vstack((M0, np.array([0, 0, 1])))
    T1 = np.vstack((M1, np.array([0, 0, 1])))
    T = np.matmul(T1, T0)
    return T[:2, :]


def random_transform(imagesize, radius):
    c = int(imagesize / 2)
    shiftrange = c - radius
    rotation_angle = np.random.uniform(0, 360)
    xshift, yshift = np.random.uniform(-shiftrange, shiftrange, 2)

    rotation_mat = cv2.getRotationMatrix2D([c, c], rotation_angle, 1)
    translation_mat = np.array([[1, 0, xshift], [0, 1, yshift]], dtype=np.float32)

    return chain_affine_transformation(rotation_mat, translation_mat)


def generate_superconducting_wire(voiddict: dict):
    radius = 400
    elementsize = 31
    variance = 0.4
    imagesize = 1000
    nbigvoid = 70
    nsmallvoid = 150
    smallvoidweights = [1, 2, 2]

    coat_in, coat_out, hexagon_in, hexagon_out = pg.generate(
        radius, elementsize, variance, imagesize
    )

    copper_mask_outer = np.zeros((1000, 1000), dtype=np.uint8)
    copper_mask_outer = cv2.fillPoly(copper_mask_outer, coat_in + hexagon_out, 255)

    fillrate = np.sum(copper_mask_outer > 0)
    if fillrate < 0.01:
        raise ValueError("Copper is too small. Increase wire radius.")
    elif fillrate < 0.05:
        warnings.warn("Copper fillrate is lower than 5%.", UserWarning)

    # BEGIN GENERATING SMALL VOID
    sum = np.sum(smallvoidweights)
    N_small_voids = np.array(smallvoidweights) / sum * nsmallvoid
    N_small_voids = N_small_voids.astype(np.int32)
    small_void_sizes = [15, 10, 7]

    void_small_mask = np.zeros_like(copper_mask_outer)

    for size, N_void in zip(small_void_sizes, N_small_voids):
        for _ in range(N_void):
            # tries 100 times to place void on copper
            is_succeed = False
            for _ in range(100):
                xpos = np.random.randint(0, imagesize - size)
                ypos = np.random.randint(0, imagesize - size)
                xcheck = xpos + int(size / 2)
                ycheck = ypos + int(size / 2)
                if copper_mask_outer[xcheck, ycheck] > 0:
                    is_succeed = True
                    break

            if not is_succeed:
                break

            vgroup = voiddict[size]
            chosenvoid = deepcopy(vgroup[np.random.randint(0, len(vgroup))])
            cv2.bitwise_and(
                chosenvoid,
                copper_mask_outer[xpos : xpos + size, ypos : ypos + size],
                chosenvoid,
            )
            cv2.bitwise_or(
                chosenvoid,
                void_small_mask[xpos : xpos + size, ypos : ypos + size],
                chosenvoid,
            )
            void_small_mask[xpos : xpos + size, ypos : ypos + size] = chosenvoid
    # END GENERATING SMALL VOID

    # BEGIN GENERATING BIG VOID
    copper_mask_inner = np.zeros((1000, 1000), dtype=np.uint8)
    copper_mask_inner = cv2.fillPoly(copper_mask_inner, hexagon_in, 255)
    void_big_mask = np.zeros_like(copper_mask_inner)

    for _ in range(nbigvoid):
        # tries 300 times to place void on copper
        is_succeed = False
        for _ in range(300):
            xpos = np.random.randint(0, imagesize - 20)
            ypos = np.random.randint(0, imagesize - 20)
            xcheck = xpos + 10
            ycheck = ypos + 10
            if copper_mask_inner[xcheck, ycheck] > 0:
                is_succeed = True
                break

        if not is_succeed:
            break

        vgroup = voiddict[20]
        chosenvoid = deepcopy(vgroup[np.random.randint(0, len(vgroup))])
        cv2.bitwise_and(
            chosenvoid,
            copper_mask_inner[xpos : xpos + 20, ypos : ypos + 20],
            chosenvoid,
        )
        cv2.bitwise_or(
            chosenvoid,
            void_big_mask[xpos : xpos + 20, ypos : ypos + 20],
            chosenvoid,
        )
        void_big_mask[xpos : xpos + 20, ypos : ypos + 20] = chosenvoid
    # END GENERATING BIG VOID
    
    # BEGIN MASKS GENERATION
    # In total we should have 6 masks (4 generated here plus 2 void masks previously generated)
    zerosmatrix = np.zeros((imagesize, imagesize), dtype=np.uint8)
    coat_mask = cv2.fillPoly(deepcopy(zerosmatrix), coat_out + coat_in, 255)
    subelement_mask = cv2.fillPoly(deepcopy(zerosmatrix), hexagon_out + hexagon_in, 255)
    void_mask = np.add(void_small_mask, void_big_mask)
    copper_mask = cv2.add(copper_mask_outer, copper_mask_inner)
    copper_mask = cv2.subtract(copper_mask, void_mask)
    background = cv2.fillPoly(deepcopy(zerosmatrix), coat_out, 255)
    # END MASKS GENERATION
    
    # BEGIN DRAWING
    finalimg = apply_mask(deepcopy(zerosmatrix), coat_mask, 20, 5)
    finalimg = apply_mask(finalimg, copper_mask, 135, 9.8)
    finalimg = apply_mask(finalimg, subelement_mask, 200, 13)
    finalimg = apply_mask(finalimg, void_mask, 50, 25)
    # finalimg = apply_mask(finalimg, void_big_mask, 50, 80)

    mask = np.stack((copper_mask, subelement_mask, void_mask), axis=-1)
    # END DRAWING


    # BEGIN TRANSLATION
    # translation_matrix = random_transform(imagesize, radius)
    rotation_angle = np.random.uniform(0, 360)
    c = int(imagesize / 2)
    translation_matrix = cv2.getRotationMatrix2D([c, c], rotation_angle, 1)
    # finalimg = cv2.warpAffine(finalimg, translation_matrix, np.shape(finalimg))
    # mask = cv2.warpAffine(mask, translation_matrix, np.shape(finalimg))
    mask_max = np.max(mask, axis=2)
    mask = (mask == np.expand_dims(mask_max, -1)) * 255
    mask_white = np.expand_dims(np.sum(mask, axis=-1) == 255 * 3, -1) * 255
    mask = mask - mask_white
    background = cv2.warpAffine(background, translation_matrix, np.shape(finalimg))
    # END TRANSLATION

    # FINALIZE AND RETURN OUTPUT
    _, background = cv2.threshold(background, 127, 255, cv2.THRESH_BINARY_INV)
    bg_color = np.random.randint(30, 200)
    background = apply_mask(background, background, 63.5, 8.1)

    finalimg = cv2.add(finalimg, background)
    return finalimg, mask


def save_wire_image(voiddict, img_dir, mask_dir, file_prefix, img_idx):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    img, mask = generate_superconducting_wire(voiddict)
    img_name = os.path.join(img_dir, f"{file_prefix}-{img_idx}.jpg")
    mask_name = os.path.join(mask_dir, f"{file_prefix}-{img_idx}.png")
    cv2.imwrite(img_name, img)
    cv2.imwrite(mask_name, mask)


def bat(folderpath: str, voiddict: dict, N_img: int = 1):
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

start = time()
# batch_generate_wire("data/train", voiddict, 1000, 20)
bat("test-normal/", voiddict, 8)
end = time() - start
print(f"Time elapsed: {end} seconds")
