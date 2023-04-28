import numpy as np
import cv2
import matplotlib.pyplot as plt
proto_file = 'model/caffe/colorization_deploy_v2.prototxt'
model_file = 'model/caffe/colorization_release_v2.caffemodel'
hull_pts = 'model/caffe/pts_in_hull.npy'
image = cv2.imread("images/grey.jpg")
net = cv2.dnn.readNetFromCaffe(proto_file,model_file)
kernel = np.load(hull_pts)
scaled = image.astype("float32") /255
lab_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = kernel.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

resized = cv2.resize(lab_img, (224, 224))
L = cv2.split(resized)[0]
L -= 50
net.setInput(cv2.dnn.blobFromImage(L))
ab_channel = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab_channel = cv2.resize(ab_channel, (image.shape[1], image.shape[0]))
L = cv2.split(lab_img)[0]
colorized = np.concatenate((L[:, :, np.newaxis], ab_channel), axis=2)
 
colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized, 0, 1)
colorized = (255 * colorized).astype("uint8")
 
image = cv2.resize(image,(640,640))
colorized = cv2.resize(colorized,(640,640))
plt.figure(figsize=[20,10])
plt.subplot(121);plt.imshow(image[...,::-1]);plt.title("orginal")
plt.subplot(122);plt.imshow(colorized[...,::-1]);plt.title("colorized")
plt.show()