import matplotlib.pyplot as plt
import numpy as np
import cv2
image=cv2.imread('images/skin.jpg')
HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
maskbgr = np.zeros((image.shape[0],image.shape[1],1), np.uint8)
for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        if image[y,x][0] >20 and image[y,x][1] >40 and image[y,x][2] > 95 and image[y,x][2]>image[y,x][1] and image[y,x][2]>image[y,x][0] and abs(image[y,x][2]-image[y,x][1])>15 :
            maskbgr[y,x]=255

min_h=0
max_h=50
min_s=0.23*255
max_s=0.68*255
min_v=0
max_v=255
lower = np.array([min_h,min_s,min_v])
upper = np.array([max_h,max_s,max_v])

maskhsv = cv2.inRange(HSV, lower, upper)

rm = cv2.bitwise_and(maskhsv, maskbgr)
result=cv2.bitwise_and(image,image, mask=rm)
plt.figure(figsize=[20,8])
plt.subplot(151);plt.imshow(image[...,::-1]);plt.title("Original");
plt.subplot(152);plt.imshow(maskhsv,cmap='gray');plt.title("maskhsv");
plt.subplot(153);plt.imshow(maskbgr,cmap='gray');plt.title("maskbgr");
plt.subplot(154);plt.imshow(rm,cmap='gray');plt.title("result_mask");
plt.subplot(155);plt.imshow(result[...,::-1]);plt.title("result");
plt.show()