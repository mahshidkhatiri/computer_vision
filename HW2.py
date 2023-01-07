import cv2
import numpy as np
image=cv2.imread("./images/amelie.jpg")
circle = np.zeros((image.shape[0],image.shape[1]), np.uint8)
cv2.circle(circle, (285,125), 80, (255,255,255), -1)
circle=cv2.add(circle,100)
png_image = np.zeros((image.shape[0],image.shape[1],4), np.uint8)
png_image[:,:,0:3] = image
png_image[:,:,3]= circle
cv2.imwrite("./images/amelie.png", png_image)

