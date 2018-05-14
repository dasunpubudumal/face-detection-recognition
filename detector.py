import numpy as np
import cv2

detector = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('recognizer/training_data.yml')
id = 0
font = cv2.FONT_HERSHEY_COMPLEX
while (True):
    ret, img = cap.read()  # ret - status variable, img - captured image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # For the classifier to work
    faces = detector.detectMultiScale(gray, 1.3, 5)  # Coordinates of the face.
    for (x, y, w, h) in faces:
        # For each face draw a rectangle around.
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        id, conf = rec.predict(gray[y:y + h, x:x + w])
        if (id == 1):
            id = "Nihal"
        if (id == 2):
            id = "Dasun"
        cv2.putText(
            img,
            str(id), (x, y + h),
            font,
            2.0, (0, 0, 255),
            lineType=cv2.LINE_AA)
    cv2.imshow('Log In', img)  # Show the window with the video.
    if cv2.waitKey(1) & 0xFF == ord(
            'q'):  # Without the wait the library would not work.
        break

cap.release()
cv2.destroyAllWindows()