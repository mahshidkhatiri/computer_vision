import cv2
import numpy as np

cap = cv2.VideoCapture("videos/coin.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([15,50,50])
    upper_red = np.array([30,255,255])
    mask= cv2.inRange(hsv, lower_red, upper_red)
    
    cv2.imshow('mask',mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        centers, radius = cv2.minEnclosingCircle(contours[0])
        centers = int(centers[0]), int(centers[1])
        radius = int(radius)
        cv2.circle(frame, centers, radius, (0,0,255), 3)
        cv2.imshow('image',frame)
    
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()