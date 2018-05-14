import os
import cv2
import numpy as np
from PIL import Image
from tkinter import messagebox

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = "dataset"


def getImagesWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    ids = []
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath)
        faceNumpy = np.array(faceImg, 'uint8')
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNumpy)
        ids.append(ID)
        cv2.imshow("Traning", faceNumpy)
        cv2.waitKey(10)
    return faces, ids


faces, ids = getImagesWithID(path)
recognizer.train(faces, np.array(ids))
messagebox.showwarning("Training", "Wait until it's trained!")
recognizer.save('recognizer/training_data.yml')
messagebox.showinfo("Success", "Successfull!")
cv2.destroyAllWindows()
