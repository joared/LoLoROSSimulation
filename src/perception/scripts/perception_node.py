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
from feature_extraction import featureAssociation, drawInfo, ThresholdFeatureExtractor
from pose_estimation import DSPoseEstimator, polygon

# remove this eventually
from pose_estimation_simulation import measurePose, CoordinateSystem, CoordinateSystemArtist

from scipy.spatial.transform import Rotation as R

def callback(msg):
    global image_msg
    image_msg = msg
    return

def main():
    global image_msg

    pixelWidth = 0.0000028
    pixelHeight = 0.0000028
    #pixelSize = 0.0009
    px = 329.864062734141
    py = 239.0201541966089
    cameraMatrix = np.array([[812.2540283203125,   0,    		        px],
                              [   0,               814.7816162109375, 	py], 
	                          [   0,     		     0,   		       	1]], dtype=np.float32)
    
    cameraMatrix[0, :] *= pixelWidth
    cameraMatrix[1, :] *= pixelHeight
    distCoeffs = np.zeros((4,1), dtype=np.float32)

    fakeRadius = 0.06
    points3D = np.array([[0, 0, -0.043, 1.]])
    points3D = np.append(points3D, polygon(rad=fakeRadius, n=4, shift=True, zShift=0), axis=0)
    nFeatures = len(points3D)

    featureExtractor = ThresholdFeatureExtractor(nFeatures=nFeatures, p=0.02)
    poseEstimator = DSPoseEstimator(cameraMatrix, distCoeffs)

    fig = plt.figure()
    axes = fig.gca(projection='3d')
    fig2 = plt.figure()
    axes2 = fig2.gca()
    cs = CoordinateSystem(scale=0.1)
    csArt = CoordinateSystemArtist(cs)
    csRef = CoordinateSystem([0,0,0], (-np.pi/2, np.pi/2, 0), scale=0.1)
    csRefArt = CoordinateSystemArtist(csRef)
    csArt.init(axes)
    csRefArt.init(axes)

    dsVel = []
    while not rospy.is_shutdown():
        if image_msg:
            try:
                imgColor = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            except CvBridgeError as e:
                print(e)
            else:
                origImg = imgColor
                imgColor = imgColor.copy()
                gray = cv.cvtColor(imgColor, cv.COLOR_BGR2GRAY)
                hist = cv.calcHist([gray], [0], None, [256], [0, 256])

                res, points = featureExtractor(gray, imgColor)
                # points are in pixels, convert points to meters
                points = np.array(points, dtype=np.float32)
                points[:,0] *= pixelWidth
                points[:,1] *= pixelHeight

                associatedPoints, featurePointsScaled, (cx,cy) = featureAssociation(points3D, points)

                for i in range(len(associatedPoints)):
                    # convert points to pixels
                    px = associatedPoints[i][0] / pixelWidth
                    py = associatedPoints[i][1] / pixelHeight
                    fpx = (featurePointsScaled[i][0]+cx) / pixelWidth
                    fpy = (featurePointsScaled[i][1]+cy) / pixelHeight
                    drawInfo(origImg, (int(px), int(py)), str(i))
                    drawInfo(origImg, (int(fpx), int(fpy)), str(i), color=(0, 0, 255))

                if len(associatedPoints) == nFeatures:
                    translationVector, rotationVector = poseEstimator.update(points3D, 
                                                                             associatedPoints, 
                                                                             np.array([[pixelWidth*2], [pixelHeight*2]]), 1)
                    
                    # show lights pose wrt to camera
                    rotMat = R.from_rotvec(rotationVector.transpose()).as_dcm()
                    transl = translationVector

                    # show camera pose wrt to lights
                    transl = np.matmul(rotMat.transpose(), -translationVector)
                    rotMat = rotMat.transpose()

                    # cancel roll and pitch
                    #ay, ax, az = R.from_dcm(rotMat).as_euler("YXZ")
                    #rotMat = R.from_euler("YXZ", (ay, 0, 0)).as_dcm() # remove roll info

                    # convert to camera coordinates (x-rgiht, y-left, z-front)
                    translation = np.matmul(csRef.rotation, transl)
                    rotation = np.matmul(csRef.rotation, np.array(rotMat))

                    #translation *= pixelSize
                    print("Range:", np.linalg.norm(translation))
                    cs.setTransform(translation, rotation) # display camera as reference frame

            cv.imshow("gray", origImg)
            axes.cla()
            axes2.cla()
            size = 0.5
            axes.set_xlim(-size, size)
            axes.set_ylim(-size, size)
            axes.set_zlim(-size, size)
            csArt.draw(axes)
            csRefArt.draw(axes)

            dsVel.append(poseEstimator.dsVelocity[2]) # z velocity
            
            axes2.plot(dsVel[-100:])
            #csArt.update()
            #csRefArt.update()
            #axes2.plot(hist)
            
        cv.waitKey(1)
        plt.pause(0.0001)

if __name__ == '__main__':
    rospy.init_node('perception_node')
    

    image_msg = None
    bridge = CvBridge()
    sub_pose = rospy.Subscriber('usb_cam/image_rect_color', Image, callback)
    pub_img = rospy.Publisher('usb_cam/image_processed', Image, queue_size=1)
    pub_img_grad = rospy.Publisher('usb_cam/image_processed_grad', Image, queue_size=1)
    pub_img_binary = rospy.Publisher('usb_cam/image_processed_bin', Image, queue_size=1)
    pub_img_binary_grad = rospy.Publisher('usb_cam/image_processed_bin_grad', Image, queue_size=1)

    main()