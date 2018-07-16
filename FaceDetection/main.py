import os
import cv2
import numpy as np
from PIL import Image

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    #classifier for simple face detection
cap = cv2.VideoCapture(0)

rec =  cv2.face.LBPHFaceRecognizer_create();
rec.read(r"C:\Users\Manish\PycharmProjects\FaceDetection\Recognizer\training_data.yml")
font = cv2.FONT_HERSHEY_COMPLEX
ID = 0

while (True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0),2)
        ID, conf = rec.predict(gray[y : y + h, x : x + w])
        name = ""
        if ID == 1:
            name = "Manish"
        elif ID == 2:
            name = "Mr. Modi"
        elif ID == 3:
            name = "Lana"

        cv2.putText(img, name, (x, y + h - 7), font, 1, (255, 0, 0))

    cv2.imshow('Face', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

