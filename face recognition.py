import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return None
    cropped_face=0
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face


def collectSamples():
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        if face_extractor(frame) is not None and ret==True:
            count += 1
            face = cv2.resize(face_extractor(frame), (300, 300))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            path_ = 'C://Users//gunja//Downloads//training//' + str(count) + '.jpg'
            cv2.imwrite(path_, face)
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Croped', face)
        else:
            print("Not found")
        if cv2.waitKey(1) == 13 or count == 1000: 
            break    
    cap.release()
    cv2.destroyAllWindows()      
    print("Collecting Samples Complete")


def modelTrain():
    data_path = 'C://Users//gunja//Downloads//training//'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    training_data, Labels = [], []
    
    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        training_data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)
    Labels = np.asarray(Labels, dtype=np.int32)
    model=cv2.face_LBPHFaceRecognizer.create()
    model.train(np.asarray(training_data), np.asarray(Labels))
    print("Model trained successfully")
    return model

model=modelTrain()

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (300, 300))
    return img, roi

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    try:
        image, face = face_detector(frame)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        results = model.predict(face)
        print(results)
        if results[1] < 500:
            confidence = int( 100 * (1 - (results[1])/400) )
            display_string = str(confidence) + '% Confident it is User'

        cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)
        if confidence > 80:
            cv2.putText(image, "Hey Gunjan", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Recognition', image )
        else:
            cv2.putText(image, "i dont know", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )
    except:
        cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.putText(image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Face Recognition', image )
        pass
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        cap.release()
        cv2.destroyAllWindows()
        break


