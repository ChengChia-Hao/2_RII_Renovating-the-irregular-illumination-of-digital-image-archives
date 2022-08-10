import numpy as np

import cv2

import imutils

import sys

input_img = cv2.imread("simple1.png")
# input_img = cv2.imread("TP1.png")
# input_img = cv2.imread("2022lena.png")

print(input_img.shape)

h = input_img.shape[0]
w = input_img.shape[1]

i = 2
countst = 0
countavi = 0

phototype = None

while (h>i and  w>i ):
    print("Now I :",i)
    print(input_img[0][0])
    print(input_img[0+i][0])
    print(input_img[0][0+i])
    print(input_img[0+i][0+i])
    a=(input_img[0][0])
    b=(input_img[0+i][0])
    c=(input_img[0][0+i])
    d=(input_img[0+i][0+i])

    a1=int(a[0])
    a2=int(a[1])
    a3=int(a[2])
    ta = a1+a2+a3
    avga=ta/3

    b1=int(b[0])
    b2=int(b[1])
    b3=int(b[2])
    tb = b1+b2+b3
    avgb=tb/3

    c1=int(c[0])
    c2=int(c[1])
    c3=int(c[2])
    tc = c1+c2+c3
    avgc=tc/3

    d1=int(d[0])
    d2=int(d[1])
    d3=int(d[2])
    td = d1+d2+d3
    avgd=td/3

    avg = (avga+avgb+avgc+avgd)/4
    print("avg = ",avg)

    i=i*2
    countst+=1
    print("ST = ",countst)
    if(avg>=200):
        countavi=countavi+1
    else: 
        countavi+=0
    print("avi = ", countavi)

DP=countavi-countst/2

if DP>=0:
    phototype=1
    print("This image over-exposure")
    print("Photo-Type = ",phototype)
elif DP<0:
    phototype=0
    print("This image under-exposure")
    print("Photo-Type = ",phototype)
else:
    print("Error")


    

