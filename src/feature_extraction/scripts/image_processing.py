#!/usr/bin/env python

def callback(msg):
    global image_msg
    image_msg = msg
    return

def publish(msg):
    pub_img.publish(msg)
    return



def main():
    global image_msg
    rate = rospy.Rate(40) #Hz

    while not rospy.is_shutdown():
        if image_msg:
            try:
                img = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            except CvBridgeError as e:
                print(e)
            image_msg = None
            #get undistorted image
            dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
            x,y,w,h = roi
            dst = dst[y:y+h, x:x+w]
            # h, w = 446, 332

            #publish it 
            publish(bridge.cv2_to_imgmsg(dst, 'bgr8'))

        rate.sleep()

if __name__ == '__main__':
    """
    import cv2
    from cv_bridge import CvBridge, CvBridgeError
    import roslib
    import rospy
    from sensor_msgs.msg import Image
    import os.path
    import glob
    #saves images from video feed
    rospy.init_node('undistort_img')

    image_msg = None
    bridge = CvBridge()

    sub_pose = rospy.Subscriber('usb_cam/image_raw', Image, callback)
    pub_img = rospy.Publisher('usb_cam/image_undist', Image, queue_size=1)

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