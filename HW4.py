import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread("images/Dark.jpg")

new_image = np.zeros(image.shape, image.dtype)

alpha = 2
beta = 30
gamma=0.35
new_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

lookUpTable = np.empty((1,256), np.uint8)

for i in range(256):
    lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)

out = cv2.LUT(image, lookUpTable)


plt.figure(figsize=[15,5])
plt.subplot(131);plt.imshow(image[...,::-1]);plt.title("Original");
plt.subplot(132);plt.imshow(new_image[...,::-1]);plt.title("Linear Output");
plt.subplot(133);plt.imshow(out[...,::-1]);plt.title("Gamma Output");