import matplotlib.pyplot as plt
import numpy as np
import cv2 
color=(0,255,0)
in_x=0
in_y=0
def click(event,x,y,flags,param):
    global color,in_x,in_y
    if event == cv2.EVENT_LBUTTONDOWN:
        in_x=x
        in_y=y
        pts = np.array([[in_x,in_y],[in_x+10,in_y+10],[in_x-10,in_y+6]], np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.fillConvexPoly(img, pts,color)
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',click)

while True:
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k == ord('b'):
        color = (255,0,0)
    elif k==ord('g'):
        color = (0,255,0)
    elif k==ord('r'):
        color = (0,0,255)
    elif k == 27:
        break
cv2.destroyAllWindows()
plt.imshow(img[...,::-1])