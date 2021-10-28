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

from feature_extraction import ThresholdFeatureExtractor

def callback(msg):
    global image_msg
    image_msg = msg
    return

def publish(msg):
    pub_img.publish(msg)
    return

def main():
    global image_msg
    rate = rospy.Rate(20) #Hz
    plt.figure()

    featureExtractor = ThresholdFeatureExtractor(nFeatures=5)
    nFeatures = 4#featureExtractor.nFeatures
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
                res1, points = featureExtractor(gray, imgColor)
                #thresholdImg = featureExtractor.imageProcessingSteps[0].img
                #openImg = featureExtractor.imageProcessingSteps[1].img
                #print("Threshold: {}".format(featureExtractor.imageProcessingSteps[0].threshold))


                
                #res2, points = extract_features_sobel(imgColor.copy(), nFeatures)
                #res3, points = extract_features_kmeans(imgColor.copy(), nFeatures)

                #publish(bridge.cv2_to_imgmsg(res, '8UC1'))
                pub_img_binary.publish(bridge.cv2_to_imgmsg(res1, '8UC1'))
                #pub_img_binary.publish(bridge.cv2_to_imgmsg(openImg, '8UC1'))
                pub_img.publish(bridge.cv2_to_imgmsg(imgColor, 'bgr8'))
                #pub_img_grad.publish(bridge.cv2_to_imgmsg(res2, 'bgr8'))
                #pub_img_grad.publish(bridge.cv2_to_imgmsg(res3, '32FC3'))
                

                #publish(bridge.cv2_to_imgmsg(res, 'bgr8'))
                #publish(bridge.cv2_to_imgmsg(res, '8UC1'))
        
        plt.pause(0.0001)
        rate.sleep()
        
    



if __name__ == '__main__':
    

    #saves images from video feed
    rospy.init_node('image_processing')

    image_msg = None
    bridge = CvBridge()
    sub_pose = rospy.Subscriber('usb_cam/image_rect_color', Image, callback)
    pub_img = rospy.Publisher('usb_cam/image_processed', Image, queue_size=1)
    pub_img_grad = rospy.Publisher('usb_cam/image_processed_grad', Image, queue_size=1)
    pub_img_binary = rospy.Publisher('usb_cam/image_processed_bin', Image, queue_size=1)
    pub_img_binary_grad = rospy.Publisher('usb_cam/image_processed_bin_grad', Image, queue_size=1)

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