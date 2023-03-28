import cv2
import numpy as np
import matplotlib.pyplot as plt
image=cv2.imread("./images/amelie.jpg")
cropped=image[60:205,210:370]
kernel_7x7 = np.ones((9, 9), np.float32) / 49
cropped_b= cv2.filter2D(cropped, -1, kernel_7x7)
image_mix=image.copy()
image_mix[60:205,210:370]=cropped_b
plt.figure(figsize=[16,5])
plt.subplot(141);plt.imshow(image[...,::-1]);plt.title("Original");
plt.subplot(142);plt.imshow(cropped[...,::-1]);plt.title("cropped ");
plt.subplot(143);plt.imshow(cropped_b[...,::-1]);plt.title("cropped with 9x9 filter");
plt.subplot(144);plt.imshow(image_mix[...,::-1]);plt.title("Mix");
plt.show()