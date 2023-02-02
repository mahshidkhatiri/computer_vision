import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread('images/hidden.png', 0)
result = cv2.equalizeHist(image)

plt.figure(figsize=[12,4])
plt.subplot(121);plt.imshow(image, cmap='gray');plt.title("Original");
plt.subplot(122);plt.imshow(result, cmap='gray');plt.title("After global histogram equalization");