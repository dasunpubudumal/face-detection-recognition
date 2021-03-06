from tkinter import *
import os


def generateDataClick():
    os.system('python3 ./util/dataset_creator.py')


def trainerClick():
    os.system('python3 ./util/trainer.py')


def detectorClick():
    os.system('python3 ./util/detector.py')


root = Tk()
root.title("Face Detector")

topFrame = Frame(root)
topFrame.pack()

bottomFrame = Frame(root)
bottomFrame.pack(side=BOTTOM)

label = Label(topFrame, text="Face Recognizer").pack(side=TOP)

datasetGeneratorBtn = Button(
    topFrame, text="1. Generate Data",
    command=generateDataClick).pack(side=LEFT)
detectBtn = Button(
    topFrame, text="3. Detect Face", command=detectorClick).pack(side=RIGHT)
trainBtn = Button(
    topFrame, text="2. Train Data", command=trainerClick).pack(side=TOP)

root.mainloop()