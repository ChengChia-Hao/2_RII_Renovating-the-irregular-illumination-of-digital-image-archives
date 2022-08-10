import cv2 

# image_result = cv2.imread("tp2-RII.png",cv2.COLOR_BGR2GRAY)
image = cv2.imread("LDL.png",cv2.COLOR_BGR2GRAY)
# image_result = cv2.imread("LDL.png",cv2.COLOR_BGR2GRAY)
# image_result = cv2.imread("LDL-r.png",cv2.COLOR_BGR2GRAY)
image_result = cv2.imread("LDL-u.png",cv2.COLOR_BGR2GRAY)
# image = cv2.imread("ex1.png",0)

width=image.shape[0]
height=image.shape[1]

widthr=image_result.shape[0]
heightr=image_result.shape[1]

# image = cv2.resize(image, (512, 512)) 
# image_result = cv2.resize(image, (512, 512))  


# widths=image_sample.shape[0]
# heights=image_sample.shape[1]
# mean = cv2.mean(image)[0]
# (mean , stddv)=cv2.meanStdDev(image)
# print ("stddv:",stddv) 
# print ("mean:",mean) 
total=0
avg=0
for i in range(width):
    for j in range(height):
        tmp= int(image[i][j][2])
        total= total+tmp 

mean=total/(width*height)

print("total:",total)
print("mean:",mean)



LDL=0
for i in range(widthr):
    for j in range(heightr):
        tmp=abs(int(image_result[i][j][2]-mean))
        LDL=LDL+tmp
        # print(i,"  :  ","LDL:",LDL)


print ("LDL Value:", LDL)