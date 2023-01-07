import cv2
import matplotlib.pyplot as plt
image=cv2.imread("./images/amelie.jpg")
plt.imshow(image[...,::-1])
cropped=image[60:205,210:370]
cv2.imwrite('./images/cropped.png',cropped)