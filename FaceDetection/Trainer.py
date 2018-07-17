import os
import cv2
import numpy as np
from PIL import Image

recognizer =  cv2.face.LBPHFaceRecognizer_create();
path = "DataSet"      

def getImageWithID(p):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        faceNp = np.array(faceImg, 'uint8')
        ID = int(os.path.split(imagePath)[-1][0])  #getting ID of the images
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("Training", faceNp)
        cv2.waitKey(10)
    return IDs, faces

Ids, faces = getImageWithID(path)
recognizer.train(faces, np.array(Ids))
recognizer.save(r'Recognizer\training_data.yml')
cv2.destroyAllWindows()


