import cv2
import matplotlib.pyplot as plt
original_image = cv2.imread('images/adad.jpg', 0)
noise_removed_image =cv2.GaussianBlur(original_image,(15,15),0)
output = cv2.adaptiveThreshold(noise_removed_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(original_image, cmap='gray');plt.title("Original");
plt.subplot(122);plt.imshow(output, cmap='gray');plt.title("Adaptive Gaussian + noise removal");