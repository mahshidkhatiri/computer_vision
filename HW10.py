import matplotlib.pyplot as plt
import numpy as np
import cv2
def find_moments(contours,image):
    for c in contours:
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.circle(image, (cx, cy), 2, (255,0,0), -1)
def moment(contour):
    M = cv2.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx,cy)
def y_cord_contour(contours):
    """Returns the X cordinate for the contour centroid"""
    if cv2.contourArea(contours) > 10:
        M = cv2.moments(contours)
        return (int(M['m01']/M['m00']))
    else:
        pass
def x_cord_contour(contours):
    """Returns the X cordinate for the contour centroid"""
    if cv2.contourArea(contours) > 10:
        M = cv2.moments(contours)
        return (int(M['m10']/M['m00']))
    else:
        pass
image=cv2.imread("images/output_adad.png")
image_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)         
_, binary_image = cv2.threshold(image_gray,127,255,cv2.THRESH_BINARY_INV)
kernel = np.ones((10, 10), np.uint8)
image_closing=cv2.morphologyEx(binary_image,cv2.MORPH_CLOSE, kernel)
contours, hierarchy = cv2.findContours(binary_image,        
                                             cv2.RETR_EXTERNAL,                 
                                             cv2.CHAIN_APPROX_SIMPLE)
find_moments(contours, image)
contours_down_to_up = sorted(contours, key = y_cord_contour, reverse = True)
boundRect = cv2.boundingRect(contours_down_to_up[-1])
cv2.rectangle(image, (int(boundRect[0]), int(boundRect[1])), \
  (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (0,255,255), 2)
contours_left_to_right = sorted(contours, key = x_cord_contour, reverse = False)
boundRect = cv2.boundingRect(contours_left_to_right[-1])
cv2.rectangle(image, (int(boundRect[0]), int(boundRect[1])), \
  (int(boundRect[0]+boundRect[2]), int(boundRect[1]+boundRect[3])), (0,0,255), 2)
plt.imshow(image)