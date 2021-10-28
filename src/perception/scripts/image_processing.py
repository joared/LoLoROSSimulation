import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R, rotation

if __name__ == "__main__":
    from feature_extraction import ThresholdFeatureExtractor, PercentageThreshold, fillContours
    thresholdFeatureExtractor = ThresholdFeatureExtractor(nFeatures=4)
    pHold = PercentageThreshold(0.15)
    
    imgOrig = cv.imread('../image_dataset/park.jpg')
    imgColor = imgOrig.copy()
    gray = cv.cvtColor(imgColor, cv.COLOR_BGR2GRAY)
    res = gray.copy()
    res = pHold.process(res)
    fillContours(res, 0)
    res, points = thresholdFeatureExtractor(res, imgColor)

    print(res)
    print(np.max(res))
    
    res = cv.bitwise_and(gray, gray, mask=res)

    hist = cv.calcHist([gray], [0], None, [256], [0, 256])
    histGrad = np.convolve(hist[:, 0], [-1, 0, 0, 1], "same")
    #histGradGrad = np.convolve(histGrad, [-1, 1], "same")
    
    for g in reversed(histGrad):
        pass

    for I in range(0, 255, 25):
        pass

    print(hist.shape)
    plt.cla()
    plt.plot(hist)
    plt.plot(histGrad)
    #plt.plot(histGradGrad)
    cv.imshow("orig", imgOrig)
    cv.imshow("color", imgColor)
    cv.imshow("result", res)
    cv.waitKey(0)
    plt.show()
