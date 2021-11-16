#!/usr/bin/env python
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import roslib
import rospy
from sensor_msgs.msg import Image
import os.path
import glob
import time
import numpy as np
import matplotlib.pyplot as plt
from feature_extraction import featureAssociation, drawInfo, ThresholdFeatureExtractor
from pose_estimation import DSPoseEstimator
from pose_estimation_utils import plotPoints, plotAxis
# remove this eventually
import sys
sys.path.append("../../simulation/scripts")
from camera import usbCamera
from feature import FeatureModel, polygon
from coordinate_system import CoordinateSystem, CoordinateSystemArtist

from scipy.spatial.transform import Rotation as R

def callback(msg):
    global image_msg
    image_msg = msg
    return

class Perception:
    def __init__(self, camera, featureModel):
        self.camera = camera
        self.featureModel = featureModel


def main():
    global image_msg

    featureModel = FeatureModel([0, 0.06], [1, 4], [False, True], [0.043, 0])
    #featureModel = FeatureModel([0.06], [4], [True], [0])
    #featureModel.features = np.array([[-0.33, -0.225, 0], [0.33, -0.225, 0], [0.33, 0.225, 0], [-0.33, 0.225, 0]])
    points3D = featureModel.features

    # points expressed as if the relative orientation between camera and featureModel is the identity matrix
    #points3D = np.array([[0, 0, 0.043, 1.]])
    #points3D = np.append(points3D, polygon(rad=0.06, n=4, shift=True, zShift=0), axis=0)
    
    # square
    #points3D = polygon(rad=0.06, n=4, shift=True, zShift=0)

    # we need to rotate the points if we want the relative orientation to be different, 
    # otherwise feature association will fail
    points3DAss = np.matmul( R.from_euler("XYZ", (0, np.pi, 0)).as_dcm(), points3D[:, :3].transpose()).transpose()    
    points3DAss = np.append(points3DAss, np.ones((points3DAss.shape[0], 1)), axis=1)

    nFeatures = len(points3D)


    camera = usbCamera
    featureExtractor = ThresholdFeatureExtractor(featureModel=featureModel, camera=camera, p=0.02, erosionKernelSize=7, maxIter=10, useKernel=False)
    poseEstimator = DSPoseEstimator(camera, ignoreRoll=False, ignorePitch=False)

    fig = plt.figure()
    axes = fig.gca(projection='3d')
    fig2 = plt.figure()
    axes2 = fig2.gca()
    axisScale = 0.2
    cs = CoordinateSystem()
    csArt = CoordinateSystemArtist(cs, scale=axisScale)
    csRef = CoordinateSystem([0,0,0], (-np.pi/2, np.pi/2, 0))
    csRefArt = CoordinateSystemArtist(csRef, scale=axisScale)
    csArt.init(axes)
    csRefArt.init(axes)

    estTranslationVec = None
    estRotationVec = None
    poseAquired = False

    dsVel = []
    startTime = time.time()
    while not rospy.is_shutdown():
        if image_msg:
            try:
                imgColor = bridge.imgmsg_to_cv2(image_msg, 'bgr8')
            except CvBridgeError as e:
                print(e)
            else:
                elapsed = time.time() - startTime
                unTouched = imgColor.copy()
                origImg = imgColor.copy()
                poseImg = imgColor.copy()
                #imgColor = imgColor.copy()
                gray = cv.cvtColor(imgColor, cv.COLOR_BGR2GRAY)
                #hist = cv.calcHist([gray], [0], None, [256], [0, 256])

                res, associatedPoints = featureExtractor(gray, imgColor, estTranslationVec, estRotationVec)

                if len(associatedPoints) == nFeatures:
                    translationVector, rotationVector, covariance = poseEstimator.update(points3D, 
                                                                                         associatedPoints, 
                                                                                         np.array([[camera.pixelWidth*2, 0], 
                                                                                                   [0, camera.pixelHeight*2]]),
                                                                                         elapsed,
                                                                                         estTranslationVec,
                                                                                         estRotationVec)

                    startTime = time.time()
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

                    print("Range:", np.linalg.norm(translation))
                    cs.setTransform(translation, rotation)

                    plotAxis(poseImg, translationVector, rotationVector, camera, points3D, 0.043)
                    plotPoints(poseImg, translationVector, rotationVector, camera, points3D, color=(0, 0, 255))
                    for p in associatedPoints:
                        x = int( p[0] / camera.pixelWidth )
                        y = int( p[1] / camera.pixelHeight )
                        radius = 2
                        cv.circle(poseImg, (x,y), radius, (255, 0, 0), 3)

                    if poseAquired:
                        print("Pose aquired")
                        estTranslationVec = translationVector
                        estRotationVec = rotationVector
            
            axes.cla()
            axes2.cla()
            size = 1
            axes.set_xlim(-size, size)
            axes.set_ylim(-size, size)
            axes.set_zlim(-size, size)
            csArt.draw(axes)
            csRefArt.draw(axes)

            dsVel.append(poseEstimator.dsVelocity[2]) # z velocity
            #axes2.plot(dsVel[-100:])
            #csArt.update()
            #csRefArt.update()
            #axes2.plot(hist)
            #cv.imshow("boundingbox", res)
            cv.imshow("original", unTouched)
            #cv.imshow("bin", featureExtractor.pHold.img)
            #cv.imshow("opened image", featureExtractor.adaOpen.img)
            #cv.imshow("image processing", imgColor)
            #cv.imshow("feature association", origImg)
            cv.imshow("pose estimation", poseImg)
            
        key = cv.waitKey(1)
        if key == ord("f"):
            poseAquired = not poseAquired
            if not poseAquired:
                estTranslationVec = None
                estRotationVec = None
        #plt.pause(0.0001)

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