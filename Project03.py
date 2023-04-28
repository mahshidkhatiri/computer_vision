import numpy as np
import cv2
import matplotlib.pyplot as plt
import dlib

def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )
    
    return dst


def warpTriangle(img1, img2, t1, t2) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = [] 
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    #img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)
    
    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)
    
    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] * ( (1.0, 1.0, 1.0) - mask )
     
    img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] = img2[r2[1]:r2[1]+r2[3], r2[0]:r2[0]+r2[2]] + img2Rect 
    


def convex(image,points):
    height, width, _ = image.shape
    hull = cv2.convexHull(np.array(points))
    cv2.drawContours(image, [hull], -1, (0,255,255), 2)
    plt.figure(figsize=[30,15])
    plt.imshow(image[...,::-1]);
    plt.show()
    return hull
def resize(image):
    height,width,_=image.shape
    max_fact=max(height,width)
    if max_fact >1000:
        scale_percent = 40
        width = int(width * scale_percent / 100)
        height = int(height * scale_percent / 100)
        dim = (width, height)
        image=cv2.resize(image,dim)
    elif max_fact <200:
        scale_percent = (1000-max_fact)//5
        width = int(width * scale_percent / 100)
        height = int(height * scale_percent / 100)
        dim = (width, height)
        image=cv2.resize(image,dim)
    return image
def rectContains(rect, point) :
    if point[0] < rect[0] :
        return False
    elif point[1] < rect[1] :
        return False
    elif point[0] > rect[2] :
        return False
    elif point[1] > rect[3] :
        return False
    return True
 
def draw_delaunay(img, delaunay_color,points ) :
    size = img.shape
    rect = (0, 0, size[1], size[0])
    subdiv = cv2.Subdiv2D(rect);
    for p in points :
        subdiv.insert(p)
    triangleList = subdiv.getTriangleList();
    size = img.shape
    r = (0, 0, size[1], size[0])
    delaunayTri = []
    
    pt = []    
        
    for t in triangleList:        
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))
        
        pt1 = (int(t[0]),int(t[1]))
        pt2 = (int(t[2]),int(t[3]))
        pt3 = (int(t[4]),int(t[5]))      
        
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            cv2.line(img, pt1, pt2, delaunay_color, 1,cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1,cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1,cv2.LINE_AA, 0)
            #Get face-points (from 68 face detector) by coordinates
            for j in range(0, 3):
                for k in range(0, len(points)):                    
                    if(abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)    
            # Three points form a triangle. Triangle array corresponds to the file tri.txt in FaceMorph 
            if len(ind) == 3:                                                
                delaunayTri.append((ind[0], ind[1], ind[2]))
        
        pt = []        
            
    plt.figure(figsize=[30,15])
    plt.imshow(img[...,::-1]);
    plt.show()
    return delaunayTri
def facedlib(image):
    frontalFaceDetector  = dlib.get_frontal_face_detector()
    faceLandmarkDetector = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    allFaces = frontalFaceDetector(imageRGB, 0)
    allFacesLandmark = []
    
    points=[]
    for k in range(0, len(allFaces)):
        faceRectangleDlib = dlib.rectangle(int(allFaces[k].left()),int(allFaces[k].top()),
              int(allFaces[k].right()),int(allFaces[k].bottom()))
        detectedLandmarks = faceLandmarkDetector(imageRGB, faceRectangleDlib)
        for p in detectedLandmarks.parts():
            points.append((p.x,p.y))
    return points
image = cv2.imread("images/mw.jpg")
image=resize(image)
image_org=image.copy()
image_org2=image.copy()
image2 = cv2.imread("images/amelie_d.jpg")
image2=resize(image2)
image2_org=image2.copy()
points=facedlib(image)
hull=convex(image,points)
points2=facedlib(image2)
hull2=convex(image2,points2)
dt=draw_delaunay( image,  (255, 255, 255),points )
dt2=draw_delaunay( image2,(255, 255, 255),points2)
alpha = 0.7


hull8U = []
hull22=[]
for i in range(0, len(hull2)):
    
    hull22.append(hull2[i][0])
hull11=[]
for i in range(0, len(hull)):
    hull8U.append((hull[i][0][0], hull[i][0][1]))
    hull11.append(hull[i][0])

for i in range(0, len(dt)):
        t1 = []
        t2 = []
        
        #get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(points[dt[i][j]])
            t2.append(points2[dt[i][j]])
        
        warpTriangle(image2_org, image_org, t2, t1)
mask = np.zeros(image.shape, dtype = image.dtype)  

cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))
plt.figure(figsize=[30,15])
plt.imshow(mask[...,::-1]);
plt.show() 
r = cv2.boundingRect(np.float32([hull11]))    
    
center = ((r[0]+int(r[2]/2), r[1]+int(r[3]/2)))  
output = cv2.seamlessClone(np.uint8(image_org), image_org2, mask, center, cv2.NORMAL_CLONE   )

cv2.imshow("Face Swapped", output)
cv2.waitKey(0)

cv2.destroyAllWindows()