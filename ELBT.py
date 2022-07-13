import numpy as np

import cv2

import imutils

import sys

# imageName = "2022lena-1.png"
imageName = "2022lena.png"


image = cv2.imread(imageName, cv2.COLOR_BGR2GRAY)

ret, bm_img= cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

cv2.imshow("Binary Edge Map", bm_img)
cv2.waitKey(0)