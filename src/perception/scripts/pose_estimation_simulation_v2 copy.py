#!/usr/bin/env python
import cv2 as cv
import numpy as np
import time
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R, rotation
from scipy.spatial.transform import Slerp
from tf.transformations import quaternion_from_matrix

from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, TransformStamped
import tf
import tf.msg
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

import sys
import os
dirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dirPath, "../../simulation/scripts"))
for p in sys.path:
    print(p)
from coordinate_system import CoordinateSystem, CoordinateSystemArtist
from feature import polygon, FeatureModel
from camera import usbCamera
from pose_estimation import DSPoseEstimator
from pose_estimation_utils import plotPoints, plotAxis, projectPoints

def vectorToPose(frameID, translationVector, rotationVector, covariance):
    rotMat = R.from_rotvec(rotationVector).as_dcm()

    # convert so that the pose is pointing in the z direction (as the light frame is defined)
    #lightRotation = R.from_euler("XYZ", (-np.pi/2, np.pi/2, 0)).as_dcm().transpose()
    #rotMat = np.matmul(rotMat, lightRotation)

    rotMatHom = np.hstack((rotMat, np.zeros((3, 1))))
    rotMatHom = np.vstack((rotMatHom, np.array([0, 0, 0, 1])))
    q = quaternion_from_matrix(rotMatHom)
    p = PoseWithCovarianceStamped()
    p.header.frame_id = frameID
    p.header.stamp = rospy.Time.now()
    (p.pose.pose.position.x, 
     p.pose.pose.position.y, 
     p.pose.pose.position.z) = (translationVector[0], 
                           translationVector[1], 
                           translationVector[2])
    p.pose.pose.orientation = Quaternion(*q)
    p.pose.covariance = list(np.ravel(covariance))

    return p

def publishVectorTransform(frameID, childFrameID, translationVector, rotationVector):
    global posePublisher, transformPublisher

    t = TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = frameID
    t.child_frame_id = childFrameID
    t.transform.translation.x = translationVector[0]
    t.transform.translation.y = translationVector[1]
    t.transform.translation.z = translationVector[2]

    rotMat = R.from_rotvec(rotationVector).as_dcm()
    rotMatHom = np.hstack((rotMat, np.zeros((3, 1))))
    rotMatHom = np.vstack((rotMatHom, np.array([0, 0, 0, 1])))
    q = quaternion_from_matrix(rotMatHom)
    t.transform.rotation = Quaternion(*q)
    transformPublisher.publish(tf.msg.tfMessage([t]))

def test2DNoiseError(camera, featureModel, sigmaX, sigmaY):
    """
    camera - Camera object
    featureModel - FeatureModel object
    pixelUncertainty - 2x2 matrix [[sigmaX, 0], [0, sigmaY]] where sigmaX and sigmaY are pixel std in x and y respectively
    """
    #img = cv.imread('../image_dataset/lights_in_scene.png')
    img = np.zeros(camera.resolution, dtype=np.int8)
    img = np.stack((img,)*3, axis=-1)

    poseEstimator = DSPoseEstimator(camera, ignorePitch=False, ignoreRoll=False)
    featurePoints = featureModel.features*1e6

    posePublisher = rospy.Publisher('light_true/pose', PoseWithCovarianceStamped, queue_size=1)
    poseNoisedPublisher = rospy.Publisher('light_noised/pose', PoseWithCovarianceStamped, queue_size=1)
    rate = rospy.Rate(20)
    i = 0
    while not rospy.is_shutdown():
        i += 1
        # True translation and rotation of the feature points wrt to camera
        #trueTrans[2] += 0.003
        trueTrans = np.array([0, 0, 1e6], dtype=np.float32)
        #trueTrans[0] += 0.003*i
        ay = np.pi#.01*i#+ 0.3*np.sin(i*0.0001-np.pi/2)
        r = R.from_euler("XYZ", (0., -ay, 0.))
        #r = R.from_euler("XYZ", (np.pi/2, 0, -np.pi/2 + ay))
        trueRotation = r.as_rotvec().transpose()
        projPoints = projectPoints(trueTrans, trueRotation, camera, featurePoints)

        # Introduce noise:
        # We systematically displace each feature point 2 standard deviations from its true value
        # 2 stds (defined by pixelUncertainty) to the left, top, right, and bottom
        #noise2D = np.random.normal(0, sigmaX, projPoints.shape)
        projPointsNoised = np.zeros(projPoints.shape)
        projPointsNoised[:, 0] = projPoints[:, 0] + np.random.normal(0, sigmaX, projPoints[:, 0].shape)
        projPointsNoised[:, 1] = projPoints[:, 1] + np.random.normal(0, sigmaY, projPoints[:, 1].shape)

        # estimate pose (translation and rotation in camera frame)
        estTranslationVec = trueTrans.copy()
        estRotationVec = trueRotation.copy()
        translationVector, rotationVector, covariance = poseEstimator.update(featurePoints, 
                                                                             projPointsNoised, 
                                                                             np.array([[4*sigmaX*sigmaX, 0.], # we set covariance as (2*std)^2 for 95% coverage
                                                                                       [0., 4*sigmaY*sigmaY]]),
                                                                             10000,
                                                                             estTranslationVec,
                                                                             estRotationVec)

        publishVectorTransform("camera1", "lights_true", 1e-6*trueTrans, trueRotation)
        posePublisher.publish( vectorToPose("camera1", 1e-6*trueTrans, trueRotation, covariance) )

        publishVectorTransform("camera2", "lights_noised", 1e-6*translationVector, rotationVector)
        poseNoisedPublisher.publish( vectorToPose("camera2", 1e-6*translationVector, rotationVector, covariance) )

        imgTemp = img.copy()
        #plotPose(img.copy(), rotation_vector, translation_vector, camera_matrix, dist_coeffs, temp3D, points_2D, transformedProj)

        #print("True trans", trueTrans)
        #print("True rot", trueRotation)
        #print("Rot", rotationVector)
        print("Trans", translationVector)

        # ground truth
        plotPoints(imgTemp, trueTrans, trueRotation, camera, featurePoints, color=(0,255,0))
        #plotAxis(imgTemp, rotation, translation, camera_matrix, dist_coeffs)

        # estimated measure points_2D and points_2D_noised should be used?: no because thats just what the PnP algotihm based the pose estimation on
        plotPoints(imgTemp, translationVector, rotationVector, camera, featurePoints, color=(0,0,255))
        plotAxis(imgTemp, translationVector, rotationVector, camera, featurePoints, scale=1e6*0.043)
        #plotAxis(imgTemp, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        #(imgTemp, rotationVector, translationVector, camera.cameraMatrix, camera.distCoeffs, pointsSolved2D, color=(0,0,255))

        imagePublisher.publish(cvBridge.cv2_to_imgmsg(imgTemp))
        #cv.imshow("Image", imgTemp)
        #cv.waitKey(10)
        rate.sleep()

if __name__ =="__main__":
    
    rospy.init_node('pose_estimation_simulation')
    transformPublisher = rospy.Publisher("/tf", tf.msg.tfMessage, queue_size=1)
    cvBridge = CvBridge()
    imagePublisher = rospy.Publisher("pose/image", Image, queue_size=1)
    camera = usbCamera
    featureModel = FeatureModel([0, 0.06], [1, 4], [False, True], [0.043, 0])
    featureModel = FeatureModel([0, 0.06], [1, 4], [False, True], [-0.1, 0])
    #featureModel.features = np.array([[], [], [], []])
    featureModel = FeatureModel([0.06], [4], [True], [0], euler=(0, np.pi, 0))
    test2DNoiseError(camera, featureModel, sigmaX=0.001*camera.pixelWidth, sigmaY=0.001*camera.pixelHeight)
    #test2DNoiseError(camera, featureModel, sigmaX=1*camera.pixelWidth, sigmaY=0.01*camera.pixelWidth)
    