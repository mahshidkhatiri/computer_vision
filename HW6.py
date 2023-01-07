import numpy as np
import cv2
color=(0,0,0)
img = np.zeros((512,512,3))
drawing = False
ix = 0
iy = 0
thickness=5
def draw(event,x,y,flags,params):
    global ix,iy,drawing,thickness,color
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix = x
        iy = y
    if event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img,(ix,iy),(x,y),color,thickness)
            ix = x
            iy = y
    if event==cv2.EVENT_LBUTTONUP:
        drawing = False
def nothing(x):
    pass
img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('image')
cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)
cv2.createTrackbar("thickness","image",1,100,nothing)
cv2.setMouseCallback("image",draw)
while True:
    cv2.imshow('image',img)
    if cv2.waitKey(1) & 0xFF ==27:
        break
    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')
    thickness=cv2.getTrackbarPos("thickness","image")
    color=(b,g,r)
cv2.destroyAllWindows()