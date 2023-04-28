import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize(video, face1,face2, image1,image2,thickness=2):
    recognizer = cv2.FaceRecognizerSF.create(
    "model/face/face_recognition_sface_2021dec.onnx","")
    face1_align = recognizer.alignCrop(image1, faces1[1][0])
    face1_feature = recognizer.feature(face1_align)
    l2_similarity_threshold = 1.128
    for idx, face in enumerate(faces2[1]):
        coords = face[:-1].astype(np.int32)
        face2_align=recognizer.alignCrop(image2, faces2[1][0])
        face2_feature = recognizer.feature(face2_align)
        l2_score = recognizer.match(face1_feature, face2_feature, cv2.FaceRecognizerSF_FR_NORM_L2)
        if l2_score <= l2_similarity_threshold:
            cv2.rectangle(video, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 255, 0), thickness)
        else:
            cv2.rectangle(video, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 0, 255), thickness)
        

    
            

detector = cv2.FaceDetectorYN.create(
    "model/face/face_detection_yunet_2022mar.onnx",
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)



image = cv2.imread("images/amelie_d.jpg")
img = image.copy()
plt.imshow(img[...,::-1]);plt.title("person picture")
plt.show()
img1Width = int(img.shape[1])
img1Height = int(img.shape[0])
img = cv2.resize(img, (img1Width, img1Height))
detector.setInputSize((img1Width, img1Height))
faces1 = detector.detect(img)
cap = cv2.VideoCapture("videos/amelie.mov")

while True:
    ret, frame = cap.read()
    detector.setInputSize((frame.shape[1], frame.shape[0]))
    faces2 = detector.detect(frame)
    if faces2[1] is not None:
        visualize(frame, faces1,faces2,image,frame)
    if not ret:
        break
    cv2.imshow('video', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
        