import cv2
import numpy as np
import matplotlib.pyplot as plt
image = np.zeros((512,512,3), np.uint8)
image[...,0][0:400,...]=cv2.add(image[...,0][0:400,...],250)
image[...,1][0:400,...]=cv2.add(image[...,1][0:400,...],200)
image[...,1][401:512,...]=cv2.add(image[...,1][401:512,...],255)
image[...,0][401:512,...]=cv2.add(image[...,0][401:512,...],120)
cv2.rectangle(image, (150,250), (350,400), (0,100,250), -1)
cv2.rectangle(image, (225,320), (275,400), (0,100,150), -1)
pts = np.array( [[150,250],[350,250],[250,110]], np.int32)
cv2.fillConvexPoly(image, pts, (0,0,255))
plt.imshow(image[...,::-1])
cv2.imwrite('./images/house.jpg',image)
