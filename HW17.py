import numpy as np
import cv2
import matplotlib.pyplot as plt

detector = cv2.FaceDetectorYN.create(
    "model/face/face_detection_yunet_2022mar.onnx",
    "",
    (320, 320),
    0.85,
    0.3,
    5000
)

image = cv2.imread("images/family.jpg")
image_org=image.copy()
height, width, _ = image.shape
detector.setInputSize((width, height))
result = detector.detect(image)

thickness=10
if result[1] is not None:
    for idx, face in enumerate(result[1]):
        coords = face[:-1].astype(np.int32)
        image[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2],...] = cv2.blur(image[coords[1]:coords[1]+coords[3],coords[0]:coords[0]+coords[2],...], (15,15))
plt.figure(figsize=[12,5])
plt.subplot(121);plt.imshow(image_org[...,::-1]);plt.title("orginal")
plt.subplot(122);plt.imshow(image[...,::-1]);plt.title("blur")
plt.show()