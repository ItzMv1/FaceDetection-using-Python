'''
            Create DataSet         - Done(9/6/18)
            Train the recognizer   - Done(12/6/18)
            Detector               - Done(14/7/18)
'''

import numpy as np
import cv2

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')    #classifier for simple face detection
cap = cv2.VideoCapture(0)

print("Enter Id for you Face = ")
id = int(input())

sammpleNum = 0;
while (True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        sammpleNum += 1
        #Here we creating every image we get in frame and labeling it with its id name

        cv2.imwrite("DataSet" + str(id) + "." + str(sammpleNum) + ".jpg", gray[y:y + h, x:x + w])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255),2)
        cv2.waitKey(100)

    cv2.imshow('Face', img)
    cv2.waitKey(1)
    if sammpleNum > 6:
        break

cap.release()
cv2.destroyAllWindows()

