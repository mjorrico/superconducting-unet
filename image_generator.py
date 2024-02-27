from scipy.spatial import Voronoi
from copy import deepcopy
import poly_generator as pg
import pyarrow.parquet as pq

import numpy as np
import cv2

coat_i, coat_o, hexa_i, hexa_o, centroids = pg.generate(300, 20, 0.2, 1000)

base = np.zeros((1000, 1000))
img = cv2.fillPoly(base, coat_i + coat_o + hexa_i + hexa_o, 255)

void50_raw = pq.read_table("void/void-50.parquet")
void_filename = void50_raw.column_names
void50 = np.stack(
    [np.array(void50_raw[file]).reshape(50, 50) for file in void_filename], axis=0
)


# BEGIN SMALL VOID MASK
base_void_small = np.zeros((1000, 1000), dtype=np.uint8)
base_void_small = cv2.fillPoly(base_void_small, coat_i + hexa_o, 255)

mask_void_small = np.zeros_like(base_void_small)

xl = np.random.randint(100, 900, 500)
yl = np.random.randint(100, 900, 500)
s = 7
for i in range(500):
    void = void50[np.random.randint(0, 7200)]
    void = cv2.resize(void, (s, s))
    _, void = cv2.threshold(void, 127, 255, cv2.THRESH_BINARY)
    x = xl[i]
    y = yl[i]
    copper = base_void_small[x : x + s, y : y + s]
    mask = mask_void_small[x : x + s, y : y + s]
    intersection = cv2.bitwise_and(copper, void)
    union = cv2.bitwise_or(intersection, mask)
    mask_void_small[x : x + s, y : y + s] = union

    # base_void_small[x : x + s, y : y + s] = new
cv2.imwrite("voidcopper.png", mask_void_small)
# END SMALL VOID MASK



# BEGIN BIG VOID MASK
base_void_big = np.zeros((1000, 1000), dtype=np.uint8)
base_void_big = cv2.fillPoly(base_void_big, hexa_i, 255)

mask_void_big = np.zeros_like(base_void_big)

xl = np.random.randint(300, 700, 100)
yl = np.random.randint(300, 700, 100)
s = 20
for i in range(100):
    void = void50[np.random.randint(0, 7200)]
    void = cv2.resize(void, (s, s))
    _, void = cv2.threshold(void, 127, 255, cv2.THRESH_BINARY)
    x = xl[i]
    y = yl[i]
    copper = base_void_big[x : x + s, y : y + s]
    mask = mask_void_big[x : x + s, y : y + s]
    intersection = cv2.bitwise_and(copper, void)
    union = cv2.bitwise_or(intersection, mask)
    mask_void_big[x : x + s, y : y + s] = union

    # base_void_small[x : x + s, y : y + s] = new
cv2.imwrite("voidelement.png", mask_void_big)
# END BIG VOID MASK

base = np.zeros((1000, 1000), dtype=np.uint8)

mask_coating = cv2.fillPoly(deepcopy(base), coat_o + coat_i, 255)
cv2.imwrite("coating.png", mask_coating)

mask_copper = cv2.fillPoly(deepcopy(base), coat_i + hexa_o + hexa_i, 255)
cv2.imwrite("copper.png", mask_copper)

mask_subelement = cv2.fillPoly(deepcopy(base), hexa_o + hexa_i, 255)
cv2.imwrite("subelement.png", mask_subelement)

c_void = np.random.randint(10, 20, (1000, 1000), dtype=np.uint8)
c_coating = np.random.randint(10, 20, (1000, 1000), dtype=np.uint8)
c_copper = np.random.randint(100, 120, (1000, 1000), dtype=np.uint8)
c_subelement = np.random.randint(190, 210, (1000, 1000), dtype=np.uint8)

def apply_mask(image, mask, low, high):
    shape = np.shape(image)[:2]
    noise = np.random.randint(low, high, shape, dtype=np.uint8)
    noise = cv2.bitwise_and(noise, noise, mask=mask)
    final_image = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    final_image = cv2.add(final_image, noise)
    return final_image

finalimg = apply_mask(deepcopy(base), mask_coating, 10, 20)
finalimg = apply_mask(finalimg, mask_copper, 170, 190)
finalimg = apply_mask(finalimg, mask_subelement, 210, 220)
finalimg = apply_mask(finalimg, mask_void_small, 50, 80)
finalimg = apply_mask(finalimg, mask_void_big, 50, 80)

rotation_matrix = cv2.getRotationMatrix2D([500, 500], np.random.uniform(0, 360), 1)
rotated_image = cv2.warpAffine(finalimg, rotation_matrix, np.shape(finalimg))
blur = cv2.blur(rotated_image,(3,3))

cv2.imwrite("finalimg.png", blur)

# img = np.zeros((1000, 1000), dtype=np.uint8)
# img = cv2.fillPoly(img, coat_o + coat_i + hexa_o + hexa_i, 255)
# img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
# img[:, :, 0] = mask_void_small
# img[:, :, 1] = mask_void_big
# cv2.imwrite("final.png", img)