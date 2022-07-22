# aAverage = []
# for i in range(10):
#     aAverage.append(2)
# print (aAverage)

# asum = sum(aAverage)
# alen = len(aAverage)
# print (asum)
# print (alen)

import numpy as np
import cv2

kernelGL1 = np.array((
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0]), dtype="float32")

kernelGL2 = np.array((
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]), dtype="float32")

kernelGL3 = np.array((
    [0,  1,  0],
    [1, -4,  1],
    [0,  1,  0]), dtype="float32")

kernelGL4 = np.array((
    [1,  1, 1],
    [1, -8, 1],
    [1,  1, 1]), dtype="float32")

kernelGL5 = np.array((
    [ 1, 2, 1],
    [ 2, 4, 2],
    [ 1, 2, 1]), dtype="float32")
    


src = np.ones((5,5),np.float32)*10
src[1,1] = 0
src[1,2] = 0
src[1,3] = 0
src[2,2] = 0
src[3,2] = 0

# for i in range(3):
#     for j in range(3):
#         src[i,j] = 10


# ======================================================================


resultGL1 = cv2.filter2D(src, dst=-1, kernel=kernelGL1, ddepth=-1, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_CONSTANT)
resultGL2 = cv2.filter2D(src, dst=-1, kernel=kernelGL2, ddepth=-1, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_CONSTANT)
resultGL3 = cv2.filter2D(src, dst=-1, kernel=kernelGL3, ddepth=-1, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_CONSTANT)
resultGL4 = cv2.filter2D(src, dst=-1, kernel=kernelGL4, ddepth=-1, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_CONSTANT)

resultGL5 = cv2.filter2D(src, dst=0, kernel=kernelGL5, ddepth=-1, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_CONSTANT)/16




cv2.imshow("src",src)
cv2.waitKey(0)
print (f"src = \n{src}")
print (f"kernelGL2 = \n{kernelGL2}")
print (f"GL2 = \n{resultGL2}")
print (f"kernelGL4 = \n{kernelGL4}")
print (f"GL4 = \n{resultGL4}")
print (f"kernelGL5 = \n{kernelGL5}")
print (f"GL5 = \n{resultGL5}")
