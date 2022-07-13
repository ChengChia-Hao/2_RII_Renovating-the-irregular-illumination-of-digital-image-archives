import numpy as np

import cv2

import imutils

import sys

# imageName = "2022lena.png"
imageName = "ex1.png"
# imageName = "AA.png"
# imageName = "TMP.jpg"
# imageName = "TP.png"
# imageName = "ELBT.png"
# imageName = "TMP.jpg"


image = cv2.imread(imageName, cv2.COLOR_BGR2GRAY)
image_gray = cv2.imread(imageName, cv2.COLOR_BGR2GRAY)
sign_img = cv2.imread(imageName, cv2.COLOR_BGR2GRAY)
LDM_img = cv2.imread(imageName, cv2.COLOR_BGR2GRAY)
Output_img = cv2.imread(imageName, cv2.COLOR_BGR2GRAY)


if image is None:

    print("Could not open or find the image")

    sys.exit()


kernel_size = 3


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
    [-1,  0,  1],
    [ 0,  1,  2]), dtype="float32")

kernel135 = np.array((
    [ 0,  1, 2],
    [-1,  0, 1],
    [-2, -1, 0]), dtype="float32")



print (kernel0)



# result0 = cv2.filter2D(image, dst=-1, kernel=kernel0, ddepth=-1, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
# result90 = cv2.filter2D(image, dst=-1, kernel=kernel90, ddepth=-1, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
# result45 = cv2.filter2D(image, dst=-1, kernel=kernel45, ddepth=-1, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
# result135 = cv2.filter2D(image, dst=-1, kernel=kernel135, ddepth=-1, anchor=(-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)

result0 = cv2.filter2D(image, dst=-1, kernel=kernel0, ddepth=-1, anchor=(1, 1), delta=0, borderType=cv2.BORDER_REPLICATE)
result90 = cv2.filter2D(image, dst=-1, kernel=kernel90, ddepth=-1, anchor=(1, 1), delta=0, borderType=cv2.BORDER_REPLICATE)
result45 = cv2.filter2D(image, dst=-1, kernel=kernel45, ddepth=-1, anchor=(1, 1), delta=0, borderType=cv2.BORDER_REPLICATE)
result135 = cv2.filter2D(image, dst=-1, kernel=kernel135, ddepth=-1, anchor=(1, 1), delta=0, borderType=cv2.BORDER_REPLICATE)

# abs0 = cv2.convertScaleAbs(result0)
# abs90 = cv2.convertScaleAbs(result90)
# abs45 = cv2.convertScaleAbs(result45)
# abs135 = cv2.convertScaleAbs(result135)

# result090= cv2.addWeighted(abs0,0.5, abs90, 0.5, 0.5)
# result45135= cv2.addWeighted(abs45, 0.5, abs135, 0.5, 0.5)
# resultavg= cv2.addWeighted(result090, 0.5, result45135, 0.5, 0.5)


result090= cv2.addWeighted(result0,0.5, result90, 0.5, 0.5)
result45135= cv2.addWeighted(result45, 0.5, result135, 0.5, 0.5)
resultavg= cv2.addWeighted(result090, 0.5, result45135, 0.5, 0.5)
ret, bm_img= cv2.threshold(resultavg, 25, 255, cv2.THRESH_BINARY)
ret, bm_img90= cv2.threshold(result090, 25, 255, cv2.THRESH_BINARY)
# ret, bm_img= cv2.threshold(resultavg, 25, 255, cv2.COLOR_BGR2GRAY)






cv2.imshow("Filter 0", result0)
cv2.imshow("Filter 90", result90)
cv2.imshow("Filter 45", result45)
cv2.imshow("Filter 135", result135)
cv2.imshow('Result avg',resultavg)
cv2.imshow("Original", image)
cv2.imshow("Binary Edge Map", bm_img)
# cv2.imshow("090 Binary Edge Map", bm_img90)
cv2.imshow("0 90 avg", result090)
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
    SOJ_img = cv2.rectangle(bm_img, (x1, y1), (x2, y2), (255, 255, 255), -1)
    
cv2.imshow("sign_img",sign_img)
cv2.imshow("SOJ_img",SOJ_img)
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
            f3=ply+1
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
            # Output_img[i][j]=(BL/1)*image_gray[i][j]
            # Output_img[i][j]=(BL/LDM_img[i][j])*image[i][j]
        else :
            Output_img[i][j]=255

cv2.imshow('Output image',Output_img)
cv2.waitKey(0)
