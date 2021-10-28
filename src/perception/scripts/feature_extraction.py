#!/usr/bin/env python
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import roslib
import rospy
from sensor_msgs.msg import Image
import os.path
import glob
import numpy as np
import matplotlib.pyplot as plt

class PercentageThreshold:
    def __init__(self, p, thresholdType=cv.THRESH_BINARY):
        """
        try cv.THRESH_OTSU?
        """
        self.p = p
        self.thresholdType = thresholdType
        self.threshold = 0 # store the last threshold that was used 
        self.img = None # store the last processed image

    def process(self, img):
        r = np.max(img) - np.min(img)
        low = int(np.max(img) - self.p*r)
        self.threshold = low
        ret, img = cv.threshold(img, low, 256, self.thresholdType)
        self.img = img
        return img

class AdaptiveErode:
    """
    Iteratively uses "opening" operations until the nr points detected in the 
    image are as close to but at least the desired nFeatures.
    The number of iterations are saved until next call to prevent unecessary reiteration.
    """
    def __init__(self, nFeatures, kernelSize=5, kernelType=cv.MORPH_ELLIPSE, startIterations=1):
        self.iterations = startIterations
        self.nFeatures = nFeatures
        self.kernel = cv.getStructuringElement(kernelType, (kernelSize,kernelSize)) 
        self.img = None

    def _findCandidatesByExclution(self, img, contours):
        """
        Find candidates by erodin the image
        """
        i = 0
        while len(contours) > self.nFeatures:
            i += 1
            imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, self.kernel, iterations=i)
            #imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_ERODE, self.kernel, iterations=i)
            _, contours, hier = cv.findContours(imgTemp, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

        if len(contours) < self.nFeatures:
            i -= 1
            imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, self.kernel, iterations=i)
            #imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_ERODE, self.kernel, iterations=i)
            _, contours, hier = cv.findContours(imgTemp, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            f = lambda cnt, img: contourAveragePixelIntensity(cnt, img)
            contours.sort(key=f, reverse=True)
            #avgPixelInt = contourAveragePixelIntensity(cnt, img)
            
        return list(contours)

    def _findCandidatesByInclusion(self, img, contours):
        
        while len(contours) < self.nFeatures and i > 0:
            i -= 1
            imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, self.kernel, iterations=i)
            #imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_ERODE, self.kernel, iterations=i)
            #imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_CROSS,(5,5)), iterations=i)
            points = _contours(imgTemp, nFeatures=None)

    def process(self, img):
        # img - binary image
        # this removes reflections effectively at close distances but reduces range
        
        imgTemp = img.copy()
        _, contours, hier = cv.findContours(imgTemp, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        i = self.iterations

        
        N = len(contours)
        contourCandidates = []
        # find candidates
        if N == self.nFeatures:
            contourCandidates = list(contours)
        elif N > self.nFeatures:
            contourCandidates = self._findCandidatesByExclution(img, contours)
        elif N < self.nFeatures:
            contourCandidates = self._findCandidatesByInclusion(img, contours)

        i = self.iterations
        while len(contours) > self.nFeatures:
            i += 1
            #imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, self.kernel, iterations=i)
            imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_ERODE, self.kernel, iterations=i)
            _, contours, hier = cv.findContours(imgTemp, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)    
        
        contours.sort(key=contourRatio, reverse=True)

        while len(points) < self.nFeatures and i > 0:
            i -= 1
            imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, self.kernel, iterations=i)
            #imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_ERODE, self.kernel, iterations=i)
            #imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_CROSS,(5,5)), iterations=i)
            points = _contours(imgTemp, nFeatures=None)

        self.iterations = i # save to we don't have to try next time
        img = imgTemp
        self.img = img
        return img

class AdaptiveOpen:
    """
    Iteratively uses "opening" operations until the nr points detected in the 
    image are as close to but at least the desired nFeatures.
    The number of iterations are saved until next call to prevent unecessary reiteration.
    """
    def __init__(self, nFeatures, kernelSize=5, kernelType=cv.MORPH_ELLIPSE, startIterations=1):
        self.iterations = startIterations
        self.nFeatures = nFeatures
        self.kernel = cv.getStructuringElement(kernelType, (kernelSize,kernelSize)) 
        self.img = None

    def process(self, img):
        # this removes reflections effectively at close distances but reduces range
        _, contours, hier = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        contours.sort(key=contourRatio, reverse=True)

        imgTemp = img.copy()
        i = self.iterations
        while len(contours) >= self.nFeatures:
            i += 1
            # doesnt really matter if open or erode is used?
            imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, self.kernel, iterations=i)
            #imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_ERODE, self.kernel, iterations=i)
            _, contours, hier = cv.findContours(imgTemp, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            contours.sort(key=contourRatio, reverse=True)

        while len(contours) < self.nFeatures and i > 0:
            i -= 1
            # doesnt really matter if open or erode is used?
            imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_OPEN, self.kernel, iterations=i)
            #imgTemp = cv.morphologyEx(img.copy(), cv.MORPH_ERODE, self.kernel, iterations=i)
            _, contours, hier = cv.findContours(imgTemp, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            contours.sort(key=contourRatio, reverse=True)

        self.iterations = i # save to we don't have to try next time
        img = imgTemp
        self.img = img
        return img

def drawInfo(img, center, text, color=(255, 0, 0)):
    # font
    font = cv.FONT_HERSHEY_SIMPLEX
    # org
    org = (center[0]-10, center[1]+10) # some displacement to make it look good
    # fontScale
    fontScale = 1
    # Line thickness of 2 px
    thickness = 2
    # Using cv2.putText() method
    image = cv.putText(img, text, org, font, 
                    fontScale, color, thickness, cv.LINE_AA)
    
    return image

def fillContours(img, ratio):
    _, contours, hier = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        r = contourRatio(cnt)
        if r < ratio:
            cv.fillPoly(img, pts=[cnt], color=(0,0,0))
        else:
            cv.fillPoly(img, pts=[cnt], color=(255,255,255))

    return img

def colorThreshold(frame):
    global image_msg
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv.bitwise_and(frame,frame, mask= mask)
    return res

def contourAveragePixelIntensity(cnt, img):
    # https://stackoverflow.com/questions/63456523/opencv-average-intensity-of-contour-figures-in-grayscale-image
    mask = np.zeros(img.shape, np.uint8)
    cv.fillPoly(mask, pts=[cnt], color=(255,255,255))
    mask_contour = mask == 255
    intensity = np.mean(img[mask_contour])
    return intensity

def cnotourCentroid(cnt):
    area = cv.contourArea(cnt)
    if area == 0:
        cx, cy = cnt[0][0]
    else:
        M = cv.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
    return cx, cy

def contourRatio(cnt):
    area = cv.contourArea(cnt)
    
    shape = "circ"
    if shape == "circ":
        (x,y),radius = cv.minEnclosingCircle(cnt)
        enclosingArea = (np.pi*radius*radius)
        if radius == 0:
            print("WTF")
            ratio = 1
        else:
            ratio = round(area/enclosingArea, 2)
    else:
        x,y,w,h = cv.boundingRect(cnt)
        enclosingArea = w*h
        ratio = round(area/enclosingArea, 2)

    return ratio

def contourAreaDistanceFromMean(contours):
    areas = np.array([cv.contourArea(cnt) for cnt in contours])
    mean = np.mean(areas)
    std = np.std(areas)
    distanceFromMean = abs(areas - mean)
    return distanceFromMean/std

def contourAreaOutliers(contours):
    maxStd = 5
    areas = np.array([cv.contourArea(cnt) for cnt in contours])
    mean = np.mean(areas)
    std = np.std(areas)
    distanceFromMean = abs(areas - mean)
    outliers = distanceFromMean > maxStd*std
    
    return [cnt for outlier, cnt in zip(outliers, contours) if not outlier]
    #return outliers

def _contours(gray, nFeatures, drawImg=None):
    mask = np.zeros(gray.shape,np.uint8)
    _, contours, hier = cv.findContours(gray.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    if drawImg is not None:
        cv.drawContours(drawImg, contours, -1, (0, 255, 0), 3)
    points = []

    contours.sort(key=contourRatio, reverse=True)
    contours = contourAreaOutliers(contours)
    #contours.sort(key=contourRatio, reverse=True)
    contours = contours[:nFeatures]

    for i, cnt in enumerate(contours):
        (x,y),radius = cv.minEnclosingCircle(cnt)
        center = (int(round(x)),int(round(y)))
        radius = int(round(radius))
        if drawImg is not None:
            cv.circle(drawImg,center,radius,(0,255,0),2)

        ratio = contourRatio(cnt)

        if True:#area < 3000:# and ratio >= 0:# and area in maxValues:
            if drawImg is not None:
                drawInfo(drawImg, (center[0]+10, center[1]-10), str(ratio))
                cv.drawContours(drawImg,[cnt],0,255,-1)


            cx, cy = cnotourCentroid(cnt)
            if drawImg is not None:
                cv.circle(drawImg, (cx, cy), int(cv.contourArea(cnt)/100), (0,0,255), 5)
            points.append((cx, cy))

    return points

def featureAssociation(featurePoints, detectedPoints):
    if len(detectedPoints) < len(featurePoints):
        print("Not enough features detected")
        return [], [], [0, 0]

    centerx, centery  = np.mean(detectedPoints, axis=0)
    xys = np.array(detectedPoints) - np.array((centerx, centery))
    maxR = np.max(np.linalg.norm(xys, axis=1))

    maxFR = np.max(np.linalg.norm(featurePoints[:, :2], axis=1)) # only x, y
    # a guess of where the features are based on the max radius of feature 
    # points and the max radius of the detected points
    featurePointsScaled = [(maxR*x/maxFR, maxR*y/maxFR) for x,y,_,_ in featurePoints]
    
    minDist = np.inf
    minDistPointIdx = None
    associatedPoints = []
    detectedPoints = [tuple(p) for p in detectedPoints]
    xys = [tuple(p) for p in xys]
    
    for i in range(len(featurePoints)):
        fx, fy = featurePointsScaled[i]
        for j in range(len(detectedPoints)):
            cx, cy = xys[j]
            d = np.sqrt( pow(cx-fx, 2) + pow(cy-fy, 2))
            if d < minDist:
                minDist = d
                minDistPointIdx = j

        p = detectedPoints[minDistPointIdx]
        associatedPoints.append(detectedPoints[minDistPointIdx])
        del xys[minDistPointIdx]
        del detectedPoints[minDistPointIdx]
        
        minDist = np.inf
        minDistPointIdx = None
    #return np.array(associatedPoints), np.array(newFeaturePoints), featurePointsScaled, (centerx, centery)
    return np.array(associatedPoints), featurePointsScaled, (centerx, centery)
    
class ThresholdFeatureExtractor:
    def __init__(self, nFeatures, p=.02):
        self.nFeatures = nFeatures
        self.pHold = PercentageThreshold(p)
        self.adaOpen = AdaptiveOpen(nFeatures, kernelSize=5)

    def __call__(self, gray, imgColor):
        img = gray.copy()
        img = self.pHold.process(img)

        _, contours, hier = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(imgColor, contours, -1, (0, 0, 255), 3)

        contours.sort(key=contourRatio, reverse=True)
        #contours = contourAreaOutliers(contours) # dont want to remove outliers here, remove noise first

        img = self.adaOpen.process(img) # removes noise
        print("Open iterations:", self.adaOpen.iterations)
        _, contoursNew, hier = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(imgColor, contoursNew, -1, (0, 255, 0), 3)

        # error: The truth value of an array with more than one element is ambiguous
        #contoursNew = np.array(contoursNew)
        #ds = contourAreaDistanceFromMean(contoursNew)
        #print(ds.shape)
        #print(contoursNew.shape)
        #contoursNew = [cnt for d, cnt in sorted(zip(ds, contoursNew))]
        contoursNew.sort(key=contourRatio, reverse=True)
        #contoursNew = contoursNew[:self.nFeatures]

        points = []
        for cntOld in contours:
            #if len(points) == self.nFeatures: # this removes potential extra contours
            #    break
            for cntNew in contoursNew:
                # Check if new contour is inside old
                area = cv.contourArea(cntNew)
                if area == 0:
                    cx, cy = cntNew[0][0]
                else:
                    M = cv.moments(cntNew)
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])

                result = cv.pointPolygonTest(cntOld, (cx,cy), False) 
                if result in (1, 0): # inside or on the contour
                    points.append((cx, cy))
                    ratio = contourRatio(cntNew)
                    cv.drawContours(imgColor,[cntNew], 0, (255, 0, 0), -1) # or maybe draw new??
                    drawInfo(imgColor, (cx+10, cy-10), str(ratio), color=(255, 0, 255))
                    #break We don't break here, the old contour might be two overlapping lights

        
        print("Npoints:", len(points))
        print("Threshold:", self.pHold.threshold)
        return img, points

if __name__ == '__main__':
    nFeatures = 4
    plt.figure()

    pHold = PercentageThreshold(0.02)
    adapOpen = AdaptiveOpen(nFeatures, kernelSize=5)
    cap = cv.VideoCapture(2)
    while True:
        _, imgColor = cap.read()
        gray = cv.cvtColor(imgColor, cv.COLOR_BGR2GRAY)

        hist = cv.calcHist([gray], [0], None, [256], [0, 256])
        plt.cla()
        plt.plot(hist)
        #res, points = extract_features_thres(imgColor.copy(), nFeatures)
        #res3, points = extract_features_kmeans(imgColor.copy(), nFeatures)
        #cv.imshow("gray", res1)
        plt.pause(0.1)
        cv.waitKey(100)