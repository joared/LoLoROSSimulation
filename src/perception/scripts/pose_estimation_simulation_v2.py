import cv2 as cv
import numpy as np
import time
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R, rotation
from scipy.spatial.transform import Slerp
from tf.transformations import quaternion_multiply

import sys
sys.path.append("../../simulation/scripts")
from coordinate_system import CoordinateSystem, CoordinateSystemArtist
from feature import polygon, FeatureModel
from pose_estimation import DSPoseEstimator
from pose_estimation_utils import plotPoints, plotAxis, projectPoints2

def projectPoints(points3D, cameraMatrix):
    # 3D points projected and converted to image coordinates (x,y in top left corner)
    points2D = np.matmul(cameraMatrix, points3D.transpose())
    points2D = points2D.transpose()
    points2D[:, 0] /= points2D[:, 2]
    points2D[:, 1] /= points2D[:, 2]
    points2D[:, 2] /= points2D[:, 2]
    #for p in points2D:
    #    p[0] /= p[2]
    #    p[1] /= p[2]
    #    p[2] /= p[2]
    return points2D[:, :2]

def displayRotation(img, objectPpoints, euler, translation, camera_matrix, dist_coeffs):
    r = R.from_euler("XYZ", euler)
    rotation = r.as_rotvec().transpose()
    nIter = 100
    
    for i in range(nIter):
        imgTemp = img.copy()
        plotAxis(imgTemp, i*rotation/nIter, translation, camera_matrix, dist_coeffs)

        # Transform and project estimated measured points
        r = R.from_rotvec((i*rotation/nIter).transpose())
        T = np.hstack((r.as_dcm(), np.array([[0], [0], [800]])))
        points3D = np.matmul(T, objectPpoints.transpose()).transpose()
        points2D = projectPoints(points3D, camera_matrix)

        plotPose(imgTemp, i*rotation/nIter, translation, camera_matrix, dist_coeffs, points2D, (255, 0, 0))
        cv.imshow("Image", imgTemp)
        cv.waitKey(10)

def pointsFromTranslEuler(translVec, eulerOrder, euler, featurePoints):
    r = R.from_euler(eulerOrder, euler)
    featureT = np.hstack((r.as_dcm(), translVec.reshape((3,1))))
    trueRotation = r.as_rotvec().transpose()
    print(trueRotation)
    # True 3D points in camera frame
    nFeatures = len(featureModel.features)
    featurePointsHomogenious = np.hstack((featurePoints, np.ones((nFeatures, 1))))
    points3D = np.matmul(featureT, featurePointsHomogenious.transpose()).transpose()
    return points3D

def test2DNoiseError(camera, featureModel, sigmaX, sigmaY):
    """
    camera - Camera object
    featureModel - FeatureModel object
    pixelUncertainty - 2x2 matrix [[sigmaX, 0], [0, sigmaY]] where sigmaX and sigmaY are pixel std in x and y respectively
    """
    img = cv.imread('../image_dataset/lights_in_scene.png')
    img = np.zeros(camera.resolution)
    img = np.stack((img,)*3, axis=-1)

    poseEstimator = DSPoseEstimator(camera)
    featurePoints = featureModel.features

    # True translation and rotation of the feature points wrt to camera
    trueTrans = np.array([0, 0, 0.2], dtype=np.float32)
    r = R.from_euler("XYZ", (0., np.pi/4, 0.))
    trueRotation = r.as_rotvec().transpose()
    projPoints = projectPoints2(trueTrans, trueRotation, camera, featurePoints)
    
    nFeatures = len(featureModel.features)
    featurePointsHomogenious = np.hstack((featurePoints, np.ones((nFeatures, 1))))
    # should the rotation be transposed? No i don't think so
    featureT = np.hstack((r.as_dcm(), trueTrans.reshape((3,1))))
    points3D = np.matmul(featureT, featurePointsHomogenious.transpose()).transpose()


    """
    # seems to be nothing wrong with this way of projecting the points
    # True 3D points in camera frame
    nFeatures = len(featureModel.features)
    featurePointsHomogenious = np.hstack((featurePoints, np.ones((nFeatures, 1))))
    featureT = np.hstack((r.as_dcm(), trueTrans.reshape((3,1))))
    points3D = np.matmul(featureT, featurePointsHomogenious.transpose()).transpose()
    print(points3D)
    # True 2D projected points, 
    projPoints2 = projectPoints(points3D, camera.cameraMatrix)
    print(projPoints2)
    """

    nIterations = 500
    for i in range(nIterations):
        # Introduce noise:
        # We systematically displace each feature point 2 standard deviations from its true value
        # 2 stds (defined by pixelUncertainty) to the left, top, right, and bottom
        print(sigmaX)
        noise2D = np.random.normal(0, sigmaX, projPoints.shape)
        projPointsNoised = projPoints + noise2D

        # estimate pose (translation and rotation in camera frame)
        estTranslationVec = trueTrans.copy()
        estRotationVec = trueRotation.copy()
        translationVector, rotationVector = poseEstimator.update(featurePoints, 
                                                                 projPointsNoised, 
                                                                 np.array([[sigmaX, 0], 
                                                                           [0, sigmaY]]),
                                                                 10000,
                                                                 estTranslationVec,
                                                                 estRotationVec)

        """# Transform and project estimated measured points
        r = R.from_rotvec(rotation_vector.transpose())
        T = np.hstack((r.as_dcm()[0], translation_vector))
        pointsSolved3D = np.matmul(T, points_3D.transpose()).transpose()
        pointsSolved2D = projectPoints(pointsSolved3D, camera_matrix)
        """
   
        imgTemp = img.copy()
        #plotPose(img.copy(), rotation_vector, translation_vector, camera_matrix, dist_coeffs, temp3D, points_2D, transformedProj)
        print("Trans", translationVector)
        print("True trans", trueTrans)
        print("Rot", rotationVector)
        print("True rot", trueRotation)
        # ground truth
        plotPoints(imgTemp, trueTrans, trueRotation, camera, featurePoints, color=(0,255,0))
        #plotAxis(imgTemp, rotation, translation, camera_matrix, dist_coeffs)

        # estimated measure points_2D and points_2D_noised should be used?: no because thats just what the PnP algotihm based the pose estimation on
        plotPoints(imgTemp, translationVector, rotationVector, camera, featurePoints, color=(0,0,255))
        #plotAxis(imgTemp, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        #(imgTemp, rotationVector, translationVector, camera.cameraMatrix, camera.distCoeffs, pointsSolved2D, color=(0,0,255))

        cv.imshow("Image", imgTemp)
        cv.waitKey(0)

if __name__ =="__main__":
    from camera import usbCamera
    camera = usbCamera
    featureModel = FeatureModel([0, 0.06], [1, 4], [False, True], [0.043, 0])
    test2DNoiseError(camera, featureModel, camera.pixelWidth*2, camera.pixelHeight*2)
    #pose2()