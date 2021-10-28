import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R, rotation
    
def gftt(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    img = cv.bitwise_and(img,img, mask= mask)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(gray,8,0.01,10)
    corners = np.int0(corners)

    for i in corners:
        x,y = i.ravel()
        cv.circle(img,(x,y),10,(0,0,255),-1)
                #publish it 
    return img

def homography(img1):

    MIN_MATCH_COUNT = 10

    sift = cv.xfeatures2d.SIFT_create()
    img2 = cv.imread('lights_in_scene.png',0) # trainImage
    img2 = cv.normalize(img2, None, 0, 255, cv.NORM_MINMAX).astype('uint8') # SIFT only accepts 8-bit
    kp2, des2 = sift.detectAndCompute(img2,None)


    # Initiate SIFT detector
    #sift = cv.SIFT()
    

    # find the keypoints and descriptors with SIFT
    img1 = cv.normalize(img1, None, 0, 255, cv.NORM_MINMAX).astype('uint8') # SIFT only accepts 8-bit
    kp1, des1 = sift.detectAndCompute(img1,None)
    

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        print(img1.shape)
        h,w,_ = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)

        img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None
    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
            singlePointColor = None,
            matchesMask = matchesMask, # draw only inliers
            flags = 2)

    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    return img3

def houghCircle(cimg):

    gray = cv.cvtColor(cimg, cv.COLOR_BGR2GRAY)

    img = cv.medianBlur(gray,5)
    #cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,8,
                                param1=50,param2=30,minRadius=10,maxRadius=30)
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            # draw the outer circle
            cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    
    return cimg
    
def histogram():
    cap = cv.VideoCapture(2)
    img = cv.imread('lights_in_scene.png',0)
    for i in range(1000):
        ret, res = cap.read()
        color = ('b','g','r')
        plt.cla()
        for i,col in enumerate(color):
            histr = cv.calcHist([res],[i],None,[256],[0,256])
            plt.plot(histr,color = col)
            plt.xlim([245,256])
        plt.pause(0.01)

    plt.show()

def extract_features_sobel(imgColor, nFeatures):
    gray = cv.cvtColor(imgColor, cv.COLOR_BGR2GRAY)
    res = gray
    #res = cv.GaussianBlur(res,(7,7),0)
    res = cv.GaussianBlur(res,(5,5),0)

    kernel = np.ones((5,5), np.uint8)
    circKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    #res = cv.morphologyEx(res.copy(), cv.MORPH_OPEN, circKernel, iterations=2)
    

    #res = cv.Canny(image=res, threshold1=10, threshold2=100)
    grad_x = cv.Sobel(res, cv.CV_64F, 1, 0, ksize=3)
    grad_y = cv.Sobel(res, cv.CV_64F, 0, 1, ksize=3)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    res = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    #res = cv.Laplacian(res, cv.CV_8UC1, 3)
    #res = cv.GaussianBlur(res,(5,5),0)
    
    #res = cv.medianBlur(res, res, 5)
    #res = cv.medianBlur(res, res, 5)
    #res = cv.medianBlur(res, 9)

    #res = cv.bitwise_not(res)
    #res, thres = percentageThres(res, p=0.7) # sobel
    #res, thres = percentageThres(res, p=0.7) # laplacian
    #res = fillContours(res, 0.7)

    kernel = np.ones((5,5), np.uint8)
    circKernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5,5))
    #res = cv.morphologyEx(res, cv.MORPH_OPEN, circKernel, iterations=2)

    
    #points = []
    #hist = cv.calcHist([res], [0], None, [256], [0, 256])
    #plt.cla()
    #plt.plot(hist)
    
    points = contours(res, nFeatures, imgColor)
    
    print("Npoints sobel:", len(points))
    return imgColor, points

def extract_features_kmeans(imgColor, nFeatures):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    gray = cv.cvtColor(imgColor, cv.COLOR_BGR2GRAY)
    res = np.float32(imgColor)
    res = cv.cvtColor(res, cv.COLOR_BGR2RGB) # Change color to RGB (from BGR) 
    # Reshaping the image into a 2D array of pixels and 3 color values (RGB) 
    print(res.shape)
    res = res.reshape((-1,3)) 
    
    #print(res)
    k = nFeatures
    _, labels, (centers) = cv.kmeans(res, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    
    imgColor = centers[labels.flatten()]
    imgColor = imgColor.reshape((480, 640, 3))
    print(imgColor.shape)
    imgColor = cv.cvtColor(imgColor, cv.COLOR_RGB2BGR)
    for c in centers:
        print(c)
        #cv.circle(imgColor, tuple(c), 10, (0,0,255), 2)

    return imgColor, centers

if __name__ == "__main__":
    pass
