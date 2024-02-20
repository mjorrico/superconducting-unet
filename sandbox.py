import skimage.exposure
import matplotlib.pyplot as plt
import numpy as np
import cv2


whites = np.random.randint(0, 255, (512, 512), dtype=np.uint8)
blur = cv2.GaussianBlur(
    whites, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT
)
stretch = skimage.exposure.rescale_intensity(blur, out_range=(0, 255)).astype(np.uint8)
thresh = cv2.threshold(stretch, 200, 255, cv2.THRESH_BINARY)[1]

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
mask1 = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel)
# mask2 = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

cv2.imshow("White", whites)
cv2.imshow("Blur", blur)
cv2.imshow("stretch", stretch)
cv2.imshow("thresh", thresh)
cv2.imshow("mask2", mask1)

cv2.waitKey(0)
cv2.destroyAllWindows()