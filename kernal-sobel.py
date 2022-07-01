import numpy as np

import cv2

import imutils

import sys

# cv2.IMREAD_COLOR為imread的預設值，此參數亦可不加。

imageName = "2022lena.png"

image = cv2.imread(imageName, cv2.IMREAD_COLOR)

#若無指定圖片則結束程式。

if image is None:

    print("Could not open or find the image")

    sys.exit()

# #縮小圖片到較適當尺寸。

# image = imutils.resize(image, height=350)

# 設定kernel size為5x5

kernel_size = 3

# 使用numpy建立 5*5且值為1/(5**2)的矩陣作為kernel。

# kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / kernel_size**3
kernel0 = np.array((
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]), dtype="float32")

kernel90 = np.array((
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]), dtype="float32")

kernel45 = np.array((
    [-2, -1,  0],
    [-1,  0,  2],
    [ 0,  1,  1]), dtype="float32")

kernel135 = np.array((
    [ 0,  1, 2],
    [-1,  0, 1],
    [-2, -1, 0]), dtype="float32")

# 顯示矩陣內容，所有值皆為0.04的5x5矩陣

print (kernel0)

# 使用cv2.filter2D進行convolute，

result0 = cv2.filter2D(image, dst=-1, kernel=kernel0, ddepth=-1, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
result90 = cv2.filter2D(image, dst=-1, kernel=kernel90, ddepth=-1, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
result45 = cv2.filter2D(image, dst=-1, kernel=kernel45, ddepth=-1, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
result135 = cv2.filter2D(image, dst=-1, kernel=kernel135, ddepth=-1, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)

abs0 = cv2.convertScaleAbs(result0)
abs90 = cv2.convertScaleAbs(result90)
abs45 = cv2.convertScaleAbs(result45)
abs135 = cv2.convertScaleAbs(result135)

result090= cv2.addWeighted(abs0,0.5, abs90, 0.5, 0.5)
result45135= cv2.addWeighted(abs45, 0.5, abs135, 0.5, 0.5)
resultavg= cv2.addWeighted(result090, 0.5, result45135, 0.5, 0.5)

ret, bm_img= cv2.threshold(resultavg, 25, 255, cv2.THRESH_BINARY)





cv2.imshow("Filter 0", result0)
cv2.imshow("Filter 90", result90)
cv2.imshow("Filter 45", result45)
cv2.imshow("Filter 135", result135)
cv2.imshow('Result avg',resultavg)
cv2.imshow("Original", image)
cv2.imshow("Binary Edge Map", bm_img)
cv2.waitKey(0)