from re import X
from turtle import down, right, up
import numpy as np
import cv2
import imutils
import sys


# imageName = "2022lena-1.png"
imageName = "2022lena.png"
# imageName = "LDI.png"
# imageName = "simple1.png"
# imageName = "under.png"

image = cv2.imread(imageName, cv2.IMREAD_COLOR)
image_gray = cv2.imread(imageName, cv2.COLOR_BGR2GRAY)
sign_img = cv2.imread(imageName, cv2.IMREAD_COLOR)
LDM_img = cv2.imread(imageName, cv2.IMREAD_COLOR)
Output_img = cv2.imread(imageName, cv2.IMREAD_COLOR)


if image is None:

    print("Could not open or find the image")

    sys.exit()


# kernel_size = 3

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
    [ -1,  2, -1],
    [  2, -4,  2],
    [ -1,  2, -1]), dtype="float32")


resultGL1 = cv2.filter2D(image, dst=-1, kernel=kernelGL1, ddepth=-1, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
resultGL2 = cv2.filter2D(image, dst=-1, kernel=kernelGL2, ddepth=-1, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
resultGL3 = cv2.filter2D(image, dst=-1, kernel=kernelGL3, ddepth=-1, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
resultGL4 = cv2.filter2D(image, dst=-1, kernel=kernelGL4, ddepth=-1, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
resultGL5 = cv2.filter2D(image, dst=-1, kernel=kernelGL5, ddepth=-1, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)

absGL1 = cv2.convertScaleAbs(resultGL1)
absGL2 = cv2.convertScaleAbs(resultGL2)
absGL3 = cv2.convertScaleAbs(resultGL3)
absGL4 = cv2.convertScaleAbs(resultGL4)
absGL5 = cv2.convertScaleAbs(resultGL5)

result1= cv2.addWeighted(absGL1,0.5, absGL2, 0.5, 0.5)
resulttmp= cv2.addWeighted(absGL4, 0.5, absGL5, 0.5, 0.5)
result2= cv2.addWeighted(resulttmp, 0.5, resultGL3, 0.5, 0.5)
resultavg= cv2.addWeighted(result1, 0.5, result2, 0.5, 0.5)

ret, bm_img= cv2.threshold(resultavg, 25, 255, cv2.THRESH_BINARY)

cv2.imshow("Original", image)
cv2.waitKey(0)

# cv2.imshow("Filter 1", resultGL1)
# cv2.imshow("Filter 2", resultGL2)
# cv2.imshow("Filter 3", resultGL3)
# cv2.imshow("Filter 4", resultGL4)
# cv2.imshow("Filter 5", resultGL5)
# cv2.waitKey(0)

cv2.imshow('Result avg',resultavg)
cv2.waitKey(0)

cv2.imshow("Binary Edge Map", bm_img)
cv2.waitKey(0)

# Signing objects

bm_img = cv2.cvtColor(bm_img, cv2.COLOR_BGR2GRAY)

contours, _2 = cv2.findContours(bm_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    cnt = contours[i]
    left    = tuple(cnt[cnt[:,:,0].argmin()][0])
    right   = tuple(cnt[cnt[:,:,0].argmax()][0])
    top     = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottom  = tuple(cnt[cnt[:,:,1].argmax()][0])
    print(i)
    print("  L = ",left[0],"  R = ",right[0],"  T = ",top[1],"  B = ",bottom[1])
    x1 = left[0]
    x2 = right[0]
    y1 = top[1]
    y2 = bottom[1] 
    sign_img = cv2.rectangle(sign_img, (x1, y1), (x2, y2), (0, 0, 0), -1)
    
cv2.imshow("sign_img",sign_img)
cv2.waitKey(0)


# Evaluating light distribution

# LDM(Px(f+k-1)) = M(Px(f-1))+[(M(Px(l+1)-M(Px(f-1)))/m]k

for i in range(len(contours)):
    cnt = contours[i]
    
    left    = tuple(cnt[cnt[:,:,0].argmin()][0])
    right   = tuple(cnt[cnt[:,:,0].argmax()][0])
    top     = tuple(cnt[cnt[:,:,1].argmin()][0])
    bottom  = tuple(cnt[cnt[:,:,1].argmax()][0])

    print(left,right,top,bottom)

    pfy = top[1]
    pfx = left[0]
    ply = bottom[1]
    plx = right[0]
    
    m = pfx-ply
    print(i+1,':','x(pfx),y(pfy),x+w(plx),y+h(ply)' , pfx, pfy, plx, ply)
    
# ``````````````````````````````````````````````````````````````````````````````````````````````
    
    for numx in range(plx-pfx):
        for numy in range(ply-pfy):
            ax = numx + pfx
            k= numy + 1
            f1=pfy+k-1
            f2=pfy-1
            f3=ply-1
            f4=pfy-1
            eq1=LDM_img[f3][ax] - LDM_img[f4][ax]
            eq2= eq1 / m
            eq = eq2 * k
            LDM_img[f1][ax] = LDM_img[f2][ax]
            resLDI = LDM_img[f1][ax]
            result = resLDI + eq
# ``````````````````````````````````````````````````````````````````````````````````````````````
   
cv2.imshow('LDI',LDM_img)
cv2.waitKey(0)
# ``````````````````````````````````````````````````````````````````````````````````````````````


# Balancing radiance


BL = 250
h, w ,t= sign_img.shape

for i in range(h):
    for j in range(w):
        if sign_img[i][j][0] == 0 :
            Output_img[i][j]=(BL/LDM_img[i][j])*image_gray[i][j]
            # Output_img[i][j]=(BL/LDM_img[i][j])*image[i][j]
        else :
            Output_img[i][j]=255

cv2.imshow('Output image',Output_img)
cv2.waitKey(0)
