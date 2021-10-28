import sys
from os.path import isfile, join
from os import listdir
import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def drawErrorCircle(img, errCircle, i, color, font=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2, delta=5):
    (x, y, r) = errCircle
    d = int(1/np.sqrt(2)*r) + delta
    org = (x+d, y+d) # some displacement to make it look good
    cv.putText(img, str(i), org, font, fontScale, color, thickness, cv.LINE_AA)
    cv.circle(img, (x,y), 1, color, 5)
    cv.circle(img, (x,y), r, color, 2)

class ImageLabeler:
    # https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
    def __init__(self, imgPath, errCircles=None):
        self.imgName = os.path.basename(imgPath)
        self.img = cv.imread(imgPath)
        self.errCircles = errCircles if errCircles else []
        self.currentPoint = None
        self.currentRadius = 10
        self.changeErrCircleIdx = None
        self.currentImg = self.img
        self.i = 0

    def _drawInfoText(self, img):
        dy = 15
        yDisplacement = 20
        infoText = "+ - increase size"
        cv.putText(img, infoText, (10,yDisplacement), 1, 1, (255, 255, 255), 1, cv.LINE_AA)
        yDisplacement += dy
        infoText = "- - decrease size"
        cv.putText(img, infoText, (10,yDisplacement), 1, 1, (255, 255, 255), 1, cv.LINE_AA)
        yDisplacement += dy
        infoText = "r - remove last label"
        cv.putText(img, infoText, (10,yDisplacement), 1, 1, (255, 255, 255), 1, cv.LINE_AA)
        yDisplacement += dy
        n = len(self.errCircles)
        if n == 1:
            infoText = "0 - change label"
            cv.putText(img, infoText, (10,yDisplacement), 1, 1, (255, 255, 255), 1, cv.LINE_AA)
            yDisplacement += dy
        elif n > 1:
            infoText = "(0-{}) - change label".format(n-1)
            cv.putText(img, infoText, (10,yDisplacement), 1, 1, (255, 255, 255), 1, cv.LINE_AA)
            yDisplacement += dy
        infoText = "n - next image"
        cv.putText(img, infoText, (10,yDisplacement), 1, 1, (255, 255, 255), 1, cv.LINE_AA)

    def draw(self):
        img = self.img.copy()
        self._drawInfoText(img)

        font = cv.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        thickness = 2
        delta = 5
        for i, (x, y, r) in enumerate(self.errCircles):
            if i == self.changeErrCircleIdx:
                color = (255, 0, 255)
            else:
                color = (0, 255, 0)
            drawErrorCircle(img, (x, y, r), i, color, font, fontScale, thickness, delta)

        if self.currentPoint:
            color = (0, 0, 255)
            idx = self.changeErrCircleIdx if self.changeErrCircleIdx is not None else len(self.errCircles)
            errCircle = self.currentPoint + (self.currentRadius,)
            drawErrorCircle(img, errCircle, idx, color, font, fontScale, thickness, delta)
            
        self.currentImg = img

    def click(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            if self.currentPoint is None:
                self.currentPoint = (x, y)
                self.currentRadius = 10
            else:
                p = self.currentPoint
                for (x, y, r) in self.errCircles:
                    d = np.sqrt(pow(p[0]-x, 2) + pow(p[1]-y, 2))
                    if d < r+self.currentRadius:
                        if self.changeErrCircleIdx is not None and (x,y,r) == self.errCircles[self.changeErrCircleIdx]:
                            # okay to overlap
                            pass
                        else:
                            print("circles are overlapping")
                            break
                else:
                    if self.changeErrCircleIdx is not None:
                        self.errCircles[self.changeErrCircleIdx] = self.currentPoint + (self.currentRadius,)
                        print("changed error circle {}".format(self.changeErrCircleIdx))
                        self.changeErrCircleIdx = None
                    else:
                        self.errCircles.append(self.currentPoint + (self.currentRadius,))
                        print("added error circle")
        
        elif event == cv.EVENT_MOUSEMOVE:
            self.currentPoint = (x, y)

        # check to see if the left mouse button was released
        elif event == cv.EVENT_LBUTTONUP:
            pass
        

    def label(self):
        cv.namedWindow(self.imgName)
        cv.setMouseCallback(self.imgName, self.click)
        while True:
            self.draw()
            # display the image and wait for a keypress
            cv.imshow(self.imgName, self.currentImg)
            key = cv.waitKey(1) & 0xFF

            if key == ord("+"): # arrow up
                print("increased size")
                self.currentRadius += 1

            elif key == ord("-"): # arrow down
                print("decreased size")
                self.currentRadius = max(self.currentRadius-1, 0)

            elif key == ord("r"):
                self.errCircles = self.errCircles[:-1]
                self.changeErrCircleIdx = None

            elif key in map(ord, map(str, range(10))):
                idx = key-48
                if idx < len(self.errCircles):
                    print("changing label {}".format(idx))
                    self.changeErrCircleIdx = idx
                elif idx == len(self.errCircles):
                    self.changeErrCircleIdx = None

            elif key == ord("n"):
                break

            else:
                pass
                #print("pressed {}".format(key))
        cv.destroyAllWindows()
        return self.errCircles


def getImagePaths(path):
    # https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    imagePaths = []
    for f in listdir(path):
        filepath = join(path, f)
        filename, file_extension = os.path.splitext(filepath)
        if file_extension in (".jpg", ".png"):
            imagePaths.append(filepath)

    return imagePaths

def saveLabeledImages(datasetPath, labelFile, labeledImgs):
    labelFile = join(datasetPath, labelFile)
    with open(labelFile, "w") as f:
        for imgFile in labeledImgs:
            if labeledImgs[imgFile]:
                labelsText = ["({},{},{})".format(x,y,r) for x,y,r in labeledImgs[imgFile]]
                f.write("{}:{}\n".format(imgFile, labelsText))

def readLabeledImages(datasetPath, labelFile):
    labelFile = join(datasetPath, labelFile)
    labeledImgs = {}
    with open(labelFile, "r") as f:
        for line in f:
            imgPath, labels = line.split(":")
            labels = labels.strip()[1:-1] # remove []
            labels = labels.split(", ")
            labels = [tuple(map(int, s[2:-2].split(","))) for s in labels]
            labeledImgs[imgPath] = labels

    return labeledImgs

def labelImages(datasetPath, labelFile):
    imgPaths = getImagePaths(datasetPath)
    labeledImgs = readLabeledImages(datasetPath, labelFile)
    
    unLabeledImgPaths = [imgPath for imgPath in imgPaths if imgPath not in labeledImgs]
    if unLabeledImgPaths:
        # First label unlabeled images
        for imgPath in unLabeledImgPaths:
            if imgPath not in labeledImgs:
                labeler = ImageLabeler(imgPath)
                labels = labeler.label()
                labeledImgs[imgPath] = labels

    else:
        # Then label already labeled images
        for imgPath in imgPaths:
            labeler = ImageLabeler(imgPath, labeledImgs[imgPath])
            labels = labeler.label()
            labeledImgs[imgPath] = labels

    saveLabeledImages(datasetPath, labelFile, labeledImgs)

def testFeatureExtractor(featExtClass, datasetPath, labelFile):
    imgPaths = getImagePaths(datasetPath)
    labeledImgs = readLabeledImages(datasetPath, labelFile)
    
    for imgPath in labeledImgs:
        img = cv.imread(imgPath)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        labels = labeledImgs[imgPath]
        nFeatures = len(labels)

        featExt = featExtClass(nFeatures, p=0.1)
        _, points = featExt(gray, img.copy())
        #_, points = featExt(gray, img)

        for i, errCircle in enumerate(labels):
            drawErrorCircle(img, errCircle, i, color=(0, 255, 0))

        classifications = {c: [] for c in labels}
        classifiedPoints = []
        for errCircle in labels:
            pointsInCircle = []
            for p in points:
                # check if inside errCircle    
                x1, y1, r = errCircle
                x2, y2 = p
                d = np.sqrt(pow(x2-x1, 2) + pow(y2-y1, 2))
                if d < r:
                    pointsInCircle.append(p)
                    classifiedPoints.append(p)
            classifications[errCircle] = pointsInCircle
            if len(pointsInCircle) == 1:
                cv.circle(img, pointsInCircle[0], 1, (255, 0, 0), 2)
            elif len(pointsInCircle) > 1:
                for p in pointsInCircle:
                    cv.circle(img, p, 1, (0, 140, 255), 2)

        for p in points:
            if p not in classifiedPoints:
                cv.circle(img, p, 1, (0, 0, 255), 2)

        infoText = "{}/{}".format(len(classifiedPoints), len(points))
        cv.putText(img, infoText, (10, 10), font, fontScale, color, thickness, cv.LINE_AA)
        cv.imshow("image", img)
        cv.waitKey(0)

if __name__ == "__main__":
    sys.path.append("scripts")
    from feature_extraction import ThresholdFeatureExtractor

    datasetPath = "image_dataset"
    labelFile = "labels.txt"
    labelImages(datasetPath, labelFile)
    #testFeatureExtractor(ThresholdFeatureExtractor, datasetPath, labelFile)