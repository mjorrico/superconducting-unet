from scipy.spatial import Voronoi
from copy import deepcopy
import poly_generator as pg

import numpy as np
import cv2


coat_i, coat_o, h_out, h_in, k = pg.generate(300, 20, 0.6, 1000)

img = np.zeros((1000, 1000))
cv2.fillPoly(img, coat_i + coat_o + h_out + h_in, 255)
cv2.imwrite("img.png", img)