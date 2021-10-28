#!/usr/bin/env python
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import roslib
import rospy
from sensor_msgs.msg import Image
import numpy as np

def callback(msg):
    global image_msg
    image_msg = msg
    return

def getChessboard(img, columns, rows):

    foundChessboard, foundCorners = cv.findChessboardCorners(img, (columns, rows),
                                                            flags=cv.CALIB_CB_FAST_CHECK)
    return foundChessboard, foundCorners

def calcPixelSize(p1, p2, squareSize):
    # pixel deltas
    dxp = abs(p1[0]- p2[0])
    dyp = abs(p1[1]- p2[1])

    theta = np.arctan(dyp/dxp)
    w = squareSize * np.cos(theta)/dxp
    h = squareSize * np.sin(theta)/dyp

    return w, h

def loop():
    global image_msg

    squareSize=0.021

    columns = 8
    rows = 6

    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        if image_msg:
            try:
                img = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            except CvBridgeError as e:
                print(e)
            else:
                foundChessboard, foundCorners = getChessboard(img, columns, rows)
                print(foundChessboard)
                if foundCorners is not None and len(foundCorners) == rows*columns:
                    print(len(foundCorners))
                    for c, in foundCorners:
                        c = tuple(c)
                        cv.circle(img, c, 1, (0, 0, 255), 2)
                    print(foundCorners.shape)
                    for c1, c2 in zip(foundCorners, foundCorners[1:]):
                        c1 = tuple(c1[0])
                        c2 = tuple(c2[0])
                        cv.line(img, c1, c2, (255, 0, 0), 2)
                        break

                    c1 = tuple(foundCorners[0][0])
                    c2 = tuple(foundCorners[1][0])
                    w, h = calcPixelSize(c1, c2, squareSize)
                    print("Width:", w)
                    print("Height:", h)
                    print(c1)
                    print(c2)

                    for r in range(rows):
                        for c in range(columns):
                            pass #w, h = calcPixelSize(c1, c2, squareSize)

                cv.imshow("image", img)
                cv.waitKey(1)
        rate.sleep()

if __name__ == "__main__":
    rospy.init_node('chessboard')

    image_msg = None
    bridge = CvBridge()
    sub_pose = rospy.Subscriber('usb_cam/image_rect_color', Image, callback)

    loop()