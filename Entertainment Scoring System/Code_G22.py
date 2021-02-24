# Mini Project Code
# Topic: Entertainment scoring system based on facial expression detection
# Group: 22

import dlib
import cv2

#Function to draw rectangles over detected faces

def rect_to_box(im, rect, i):
    
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    
    cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 1)
    startX = x
    startY = y - 15 if y - 15 > 15 else y + 15
    cv2.putText(im, str(i + 1), (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
   
    return im

# Function to draws 68 facial landmarks of each detected face on the image

def annotate_landmarks(im, rect):
    
    landmarks = [[p.x, p.y] for p in predictor(im, rect).parts()]
    
    for i, point in enumerate(landmarks):
        pos = (point[0], point[1])
        cv2.putText(im, str(i + 1), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255))
        cv2.circle(im, pos, 2, color=(0, 255, 255))
    
    return im, landmarks

#Function to predict the facial expression of each detected face in an image

def predict_facial_expression(landmarks, expressions):
    
    ux,uy = landmarks[62]
    lx,ly = landmarks[60]
    rx,ry = landmarks[64]
    dx,dy = landmarks[66]
    
    slope = float(ry - ly) / float(rx - lx)
    intercept = slope * (-1) * rx + ry
    dist1 = (slope * ux - uy + intercept) / ((1 + slope ** 2) ** 0.5)
    dist2 = (slope * dx - dy + intercept) / ((1 + slope ** 2) ** 0.5)
    left_to_right = ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5
    
    if(dist1 > 0 and dist2 > 0):
        expressions[1] += 1
    elif(dist1 < 0 and dist2 < 0):
        expressions[0] += 1
    else:
        if(dist1 > abs(dist2)):
            if((dist1 - dist2) / left_to_right > 0.15):
                expressions[4] += 1
            else:
                expressions[2] += 1
        else:
            if((dist1 - dist2) / left_to_right < 0.15):
                expressions[2] += 1
            elif((dist1 - dist2) / left_to_right < 0.6):
                expressions[0] += 1
            else:
                expressions[3] += 1
                
    return expressions

#Reading image and convering into gray

image = cv2.imread("image.jpg")
dim = image.shape
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Creating predicatable object of trained model to detect 68 facial landmarks on test images

PREDICTOR_PATH = "C:/Users/Username/Anaconda3/envs/py27/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)

#Detecting and returning a list of objects which outline every face in the image

detector = dlib.get_frontal_face_detector()
rects, scores, idx = detector.run(gray, 1, 0.25)

#Creating some clones of images to show output/result

clone = image.copy()
clone1 = image.copy()
clone2 = image.copy()
clone2 = cv2.resize(clone2, (500, 500 * dim[0] / dim[1]), interpolation = cv2.INTER_LINEAR)

#Expresions matrix to store count of different types of facial expressions

expressions = [0, 0, 0, 0, 0]

for i,rect in enumerate(rects):
    clone = rect_to_box(clone, rect, i)
    clone1, landmarks = annotate_landmarks(clone1, rect)
    expressions = predict_facial_expression(landmarks, expressions)

cv2.putText(clone2, "Happy: " + str(expressions[0]), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
cv2.putText(clone2, "Sad: " + str(expressions[1]), (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
cv2.putText(clone2, "Normal: " + str(expressions[2]), (5, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
cv2.putText(clone2, "Surprised: " + str(expressions[3]), (5, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
cv2.putText(clone2, "Terrified: " + str(expressions[4]), (5, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

cv2.imshow("Original Image", image)
cv2.imshow("Image with detected faces", clone)
#cv2.imwrite("image1.jpg", clone)
cv2.imshow("Image with 68 facial landmarks", clone1)
#cv2.imwrite("image2.jpg", clone1)
cv2.imshow("Image with predicted number of facial expressions", clone2)
#cv2.imwrite("image3.jpg", clone2)
cv2.waitKey(0)
cv2.destroyAllWindows()