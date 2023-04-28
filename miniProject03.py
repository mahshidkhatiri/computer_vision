import cv2
import matplotlib.pyplot as plt
def drawRectangle(frame, box,class_name):
    p1 = (int(bbox[0]), int(box[1]))
    p2 = (int(bbox[0] + bbox[2]), int(box[1] +box[3]))
    cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
    cv2.putText(frame, class_name , (int(box[0]),int(box[1])-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
model = cv2.dnn.readNetFromTensorflow('model/tensorflow/frozen_inference_graph.pb', 'model/tensorflow/ssd_mobilenet_v2_coco_2018_03_29.pbtxt')

with open('model/tensorflow/object_detection_classes_coco.txt', 'r') as f:
    class_names = f.read().split('\n')
video = cv2.VideoCapture("videos/animal.mp4")
ret,frame = video.read() 
image_height, image_width, _ = frame.shape


tracker = cv2.legacy_TrackerBoosting.create()
ditectted=False
while True:
    if not ditectted:
        model.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))
        output = model.forward()
        for detection in output[0,0,:,:]:
            score = float(detection[2])
            if score > 0.3:
                ditectted=True
                class_id = detection[1]
                class_name = class_names[int(class_id)-1]
                left = detection[3] * image_width
                top = detection[4] * image_height
                right = detection[5] * image_width
                bottom = detection[6] * image_height
                
                cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=1)
                cv2.putText(frame, class_name, (int(left), int(top - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
    else:
        bbox = (int(left), int(top),int(right)-int(left), int(bottom)-int(top))
        ok = tracker.init(frame, bbox)
        tm = cv2.TickMeter()
        tm.start()
        ok, bbox = tracker.update(frame)
        tm.stop()
        if ok:
            drawRectangle(frame, bbox,class_name)
        else :
            cv2.putText(frame, "Tracking failure detected", (80,140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        
        if not ok:
            break
    cv2.imshow('video', frame)
    if not ret:
        break
    ret, frame = video.read()
    if cv2.waitKey(1) & 0xFF == 27:
        break

    