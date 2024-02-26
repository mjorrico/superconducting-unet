import numpy as np
import glob
import cv2

rawvoid_250 = []
rawvoid_007 = []
rawvoid_010 = []
rawvoid_020 = []
voidpath = "void/*.png"
for imgpath in glob.glob(voidpath):
    raw = cv2.imread(imgpath)
    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    _, raw = cv2.threshold(raw, 127, 255, cv2.THRESH_BINARY_INV)
    rawvoid_250.append(raw)

    img = cv2.resize(raw, (20, 20))
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    rawvoid_020.append(img)

    img = cv2.resize(raw, (10, 10))
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    rawvoid_010.append(img)

    img = cv2.resize(raw, (7, 7))
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    rawvoid_007.append(img)

center = [int(i/2) for i in np.shape(rawvoid_250[0])]
print(np.shape(rawvoid_007[0]))
rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1)
rotated_image = cv2.warpAffine(rawvoid_250[0], rotation_matrix, np.shape(rawvoid_250[0]))
print(np.shape(rawvoid_250[0]))
print(np.shape(rotated_image))
cv2.imwrite("rotate.jpg", rotated_image)


def impose_void(big_image: np.ndarray, small_image: np.ndarray):
    base = np.zeros_like(big_image)

    b_height, b_width = np.shape(big_image)
    s_height, s_width = np.shape(small_image)
    s_height_half = np.int32(s_height / 2)
    s_width_half = np.int32(s_width / 2)
    height_range = b_height - s_height
    width_range = b_width - s_width

    for i in range(100):  # tries 100 times to impose
        x = np.random.randint(0, height_range)
        y = np.random.randint(0, width_range)
        if big_image[x + s_height_half, y + s_width_half] == 255:
            patch = base[x : x + s_height, y : y + s_width]
            base[x : x + s_height, y : y + s_width] = cv2.bitwise_and(patch, small_image)
    
