import cv2
img = cv2.imread('sc.png')

img = 255-img 

cv2.imshow('SC', img)
cv2.waitKey(0)
cv2.destroyAllWindows()