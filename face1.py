import numpy as np
import cv2  
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml');
  
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')  
  

cam = cv2.VideoCapture(1);
while(True):  
  
    ret,img =cam.read();  
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    faces=faceDetect.detectMultiScale(gray,1.3,5); 
  
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w] 
        eyes = eye_cascade.detectMultiScale(roi_gray)  
        for (ex,ey,ew,eh) in eyes: 
           cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2) 
    cv2.imshow('Face',img);  
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release() 
cv2.destroyAllWindows()



