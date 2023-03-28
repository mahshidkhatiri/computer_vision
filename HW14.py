import cv2
import numpy as np
import matplotlib.pyplot as plt

rotation_amount_degree = int(input("how much you want to rotate your picture: "))
theta = rotation_amount_degree * np.pi / 180.0
image = cv2.imread('images/amelie.jpg')
height, width, _ = image.shape
T = np.float32([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]) #Rotate
final_T = np.zeros((2,3))
final_T[:,:-1] = T
final_T[0,2]=10
result = cv2.warpAffine(image, final_T, (width, height))
plt.subplot(121);plt.imshow(image[...,::-1]);plt.title("image");
plt.subplot(122);plt.imshow(result[...,::-1]);plt.title("result");
plt.show()