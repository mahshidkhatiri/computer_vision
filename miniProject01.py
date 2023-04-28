import cv2
protoFile = "model/caffe/pose_deploy_linevec.prototxt"
weightsFile = "model/caffe/pose_iter_440000.caffemodel"
nPoints = 15
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
def DrawSkeleton(imSkeleton,points,msg):
    if point[0]!= None:
        cv2.putText(imSkeleton, msg, (points[0][0],points[0][1]-30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200,200,250), 1)
    else:
        cv2.putText(imSkeleton, msg, (50,50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200,200,250), 1)
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
    
        if points[partA] and points[partB]:
            cv2.line(imSkeleton, points[partA], points[partB], (255, 255,0), 2)
            cv2.circle(imSkeleton, points[partA], 8, (255, 0, 0), thickness=-1, lineType=cv2.FILLED)
    return imSkeleton
def recognizer(points):
    thershold_siting=5
    if points[8]!=None and points[11]!=None:
        if points[9][1]-points[8][1]<thershold_siting and points[12][1]-points[11][1]<thershold_siting:
            msg="sitted down"
        elif points[9][1]-points[8][1]<thershold_siting and (points[12]==None or points[11]==None):
            msg="sitted down"
        elif points[11][1]-points[12][1]<thershold_siting and (points[8]==None or points[9]==None):
            msg="sitted down"
        elif points[9][1]-points[8][1]<thershold_siting+2 or points[12][1]-points[11][1]<thershold_siting+2:
            msg="half sitted"
        elif points[9][1]-points[8][1]<thershold_siting:
            msg="maybe running"
        else:
            msg="stood up"
    else:
        msg="Can't Recognize"
    return msg
cap = cv2.VideoCapture("videos/poses.mov")
while True:
    ret,image= cap.read()
    if ret :
        height, width, _ = image.shape
        blob = cv2.dnn.blobFromImage(image, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(blob)
        output = net.forward()
        
        scaleX = width / output.shape[3]
        scaleY = height / output.shape[2]
        
        # Empty list to store the detected keypoints
        points = []
        points_ws=[]
        # Treshold 
        threshold = 0.1
        for i in range(nPoints):
            # Obtain probability map
            probMap = output[0, i, :, :]
            
            # Find global maxima of the probMap.
            _, prob, _, point = cv2.minMaxLoc(probMap)
            
            # Scale the point to fit on the original image
            x = scaleX * point[0]
            y = scaleY * point[1]
            
            if prob > threshold : 
                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
                points_ws.append((point[0],point[1]))
            else :
                points.append(None)
                points_ws.append(None)
    
        msg=recognizer(points_ws)
        image=DrawSkeleton(image,points,msg)
    else:
        break
    cv2.imshow('video', image)
    if cv2.waitKey(1) & 0xFF == 27:
        break

