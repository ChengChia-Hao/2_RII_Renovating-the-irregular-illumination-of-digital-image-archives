import cv2

img = cv2.imread("textth.png" ,cv2.COLOR_BGR2GRAY)

ret, th_img= cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)

cv2.imshow("th",th_img)
cv2.waitKey(0)
