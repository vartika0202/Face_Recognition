import numpy as np
import cv2  
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')  

cam = cv2.VideoCapture(1)

rec=cv2.createLBPHFaceRecognizer();
rec.load("recognizer\\trainningData.yml")
id=0
font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
while(True):  
  
    ret, img = cam.read();  
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    faces = faceDetect.detectMultiScale(gray, 1.3, 5); 
  
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])

        roi_gray = gray[y:y+h, x:x+w] 
        roi_color = img[y:y+h, x:x+w] 
        eyes = eye_cascade.detectMultiScale(roi_gray)  
        for (ex,ey,ew,eh) in eyes: 
           cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,127,255),2)
        c=100-conf
        print(c)
       # if(conf>45):
        #    id="unknown"
        #else:
        if(id==1):
            id="Aishwariya Rai"
        elif(id==2):
            id="Katrina Kaif"
        elif(id==3):
            id="Kareena Kapoor"
        cv2.cv.PutText(cv2.cv.fromarray(img),str(id),(x,y+h),font,255);
    cv2.imshow('Face',img)  
    if(cv2.waitKey(1)==ord('q')):
        break;
cam.release() 
cv2.destroyAllWindows()
