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
from feature_extraction import percentageThres, contours, erosionDilute

def callback(msg):
    global image_msg
    image_msg = msg
    return

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

def extract_features_thres(imgColor, nFeatures):
    res = cv.cvtColor(imgColor, cv.COLOR_BGR2GRAY)
    #res = cv.medianBlur(res,5)
    #res = cv.cvtColor(res, cv.COLOR_GRAY2BGR)
    res = cv.GaussianBlur(res,(5,5),0)
    res, thres = percentageThres(res, p=0.01)
    cv.imshow("bin", res)
    print("Threshold:", thres)
    res = erosionDilute(res)
    
    pub_img_binary.publish(bridge.cv2_to_imgmsg(res, '8UC1'))
    res, points = contours(imgColor, res, nFeatures)
    
    print("Npoints:", len(points))
    return res, points

def main():
    global image_msg
    plt.figure()
    while not rospy.is_shutdown():
        if image_msg:
            try:
                imgColor = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            except CvBridgeError as e:
                print(e)
            else:
                gray = cv.cvtColor(imgColor, cv.COLOR_BGR2GRAY)
                hist = cv.calcHist([gray], [0], None, [256], [0, 256])
                plt.cla()
                plt.plot(hist)
                res, points = extract_features_thres(imgColor, 8)
                #res, points = extract_features_sobel(imgColor)

                #publish(bridge.cv2_to_imgmsg(res, '8UC1'))
                #pub_img.publish(bridge.cv2_to_imgmsg(res, 'bgr8'))
                cv.imshow("gray", res)
                print(123)

                #publish(bridge.cv2_to_imgmsg(res, 'bgr8'))
                #publish(bridge.cv2_to_imgmsg(res, '8UC1'))
        cv.waitKey(1)
        plt.pause(0.0001)



if __name__ == '__main__':
    

    #saves images from video feed
    rospy.init_node('image_processing')

    image_msg = None
    bridge = CvBridge()
    sub_pose = rospy.Subscriber('usb_cam/image_rect_color', Image, callback)
    pub_img = rospy.Publisher('usb_cam/image_processed', Image, queue_size=1)
    pub_img_binary = rospy.Publisher('usb_cam/image_processed_bin', Image, queue_size=1)

    main()
    """
    import cv2 as cv
    import numpy as np
    cap = cv.VideoCapture(2)
    while(1):
        # Take each frame
        _, frame = cap.read()
        # Convert BGR to HSV
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        # define range of blue color in HSV
        lower_blue = np.array([110,50,50])
        upper_blue = np.array([130,255,255])
        # Threshold the HSV image to get only blue colors
        mask = cv.inRange(hsv, lower_blue, upper_blue)
        # Bitwise-AND mask and original image
        res = cv.bitwise_and(frame,frame, mask= mask)
        cv.imshow('frame',frame)
        cv.imshow('mask',mask)
        cv.imshow('res',res)
        k = cv.waitKey(5) & 0xFF
        if k == 27:
            break
    cv.destroyAllWindows()
    """