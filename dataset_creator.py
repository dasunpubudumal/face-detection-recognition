import numpy as np
import cv2
from tkinter import *
from tkinter import messagebox


def submitFunc(entryId, root):
    detector = cv2.CascadeClassifier(
        'data/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    identifier = entryId.get()
    sampleNumber = 0

    while (True):
        ret, img = cap.read()  # ret - status variable, img - captured image
        gray = cv2.cvtColor(img,
                            cv2.COLOR_BGR2GRAY)  # For the classifier to work
        faces = detector.detectMultiScale(gray, 1.3,
                                          5)  # Coordinates of the face.
        for (x, y, w, h) in faces:
            sampleNumber = sampleNumber + 1
            cv2.imwrite("dataset/User." + str(identifier) + "." +
                        str(sampleNumber) + ".jpg", gray[y:y + h, x:x + w])
            # For each face draw a rectangle around.
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.waitKey(100)
        cv2.imshow('Sign Up', img)  # Show the window with the video.
        cv2.waitKey(1)

        if (sampleNumber >= 20):
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Success", "Successfully Captured")
    root.quit()


root = Tk()
root.title("ID")

topFrame = Frame(root)
topFrame.pack()

bottomFrame = Frame(root)
bottomFrame.pack()

label = Label(topFrame, text="Enter ID").pack(side=BOTTOM)
entryId = Entry(bottomFrame, text="Enter ID")
entryId.pack(side=TOP)
submitBtn = Button(bottomFrame, text="Submit")
submitBtn.bind(
    "<Button>",
    lambda event, entryId=entryId, root=root: submitFunc(entryId, root))
submitBtn.pack(side=BOTTOM)

# identifier = input("Enter ID: ")

root.mainloop()
