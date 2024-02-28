from copy import deepcopy
import poly_generator as pg
import pyarrow.parquet as pq
import glob
import os

import warnings
import numpy as np
import cv2


def apply_mask(image, mask, low, high):
    shape = np.shape(image)[:2]
    noise = np.random.randint(low, high, shape, dtype=np.uint8)
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


def __generate_superconducting_wire(
    voiddict: dict,
    radius: int = 300,
    elementsize: int = 20,
    variance: float = 0.2,
    imagesize: int = 1000,
    nbigvoid: int = 30,
    nsmallvoid: int = 100,
    smallvoidweights: list = [1, 2, 1],
):
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
    finalimg = apply_mask(deepcopy(zerosmatrix), coat_mask, 10, 20)
    finalimg = apply_mask(finalimg, copper_mask, 170, 190)
    finalimg = apply_mask(finalimg, subelement_mask, 210, 220)
    finalimg = apply_mask(finalimg, void_small_mask, 50, 80)
    finalimg = apply_mask(finalimg, void_big_mask, 50, 80)

    mask = np.stack((copper_mask, subelement_mask, void_mask), axis=-1)
    # END DRAWING

    # BEGIN TRANSLATION
    translation_matrix = random_transform(imagesize, radius)
    
    finalimg = cv2.warpAffine(finalimg, translation_matrix, np.shape(finalimg))
    mask = cv2.warpAffine(mask, translation_matrix, np.shape(finalimg))
    background = cv2.warpAffine(background, translation_matrix, np.shape(finalimg))
    # END TRANSLATION

    # FINALIZE AND RETURN OUTPUT
    _, background = cv2.threshold(background, 127, 255, cv2.THRESH_BINARY_INV)
    background = apply_mask(background, background,  0, 255)

    finalimg = cv2.blur(finalimg, (3, 3))
    finalimg = cv2.add(finalimg, background)

    return finalimg, mask, background


voiddict = read_void("void/")
img, mask, background = __generate_superconducting_wire(voiddict, 300, 20, 0.2, 1000, 20, 100, [1, 2, 2])
cv2.imwrite("test.jpg", img)
cv2.imwrite("testmask.jpg", mask)
cv2.imwrite("testbackground.jpg", background)

from time import time

start = time()
N = 1000
for i in range(N):
    print(f"Progress: {i+1}/{N}.")
    __generate_superconducting_wire(voiddict, 300, 20, 0.2, 1000, 20, 100, [1, 2, 2])
elapsed = time() - start
print(f"Time elapsed: {elapsed} seconds or {elapsed / N} seconds per image")
