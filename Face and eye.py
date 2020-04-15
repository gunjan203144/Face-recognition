import cv2
import numpy as np

face_model=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_model=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')

camera=cv2.VideoCapture(0)
# camera=cv2.VideoCapture('http://192.168.1.5:8080/video')

while True:
    ret, photo=camera.read()
    if ret==True:
        arr=face_model.detectMultiScale(photo)
        for (x,y,w,h) in arr:
            cv2.rectangle(photo,(x,y),(x+w,y+h),[0,255,0],2)
        for (x,y,w,h) in eye_model.detectMultiScale(photo):
            cv2.rectangle(photo,(x,y),(x+w,y+h),[0,0,255],2)
        faces='Detected Faces = '+str(len(arr))
        photo = cv2.putText(photo, faces, (50,50),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2, cv2.LINE_AA) 
        cv2.imshow('Leader',photo)
        # Press enter to exit 13
        # Press esc to exit 27
        if cv2.waitKey(1)==13:
            cv2.destroyAllWindows()
            camera.release()
            break