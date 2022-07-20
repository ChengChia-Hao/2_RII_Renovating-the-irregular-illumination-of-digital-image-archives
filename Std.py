import cv2 

image = cv2.imread("zzz.png",0)
# image = cv2.imread("ex1.png",0)
width=image.shape[0]
height=image.shape[1]
mean = cv2.mean(image)[0]
(mean , stddv)=cv2.meanStdDev(image)
print ("stddv:",stddv) 
print ("mean:",mean) 