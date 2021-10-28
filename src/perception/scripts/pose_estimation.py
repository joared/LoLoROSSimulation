
import cv2 as cv
import numpy as np
import time
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from scipy.spatial.transform import Rotation as R, rotation
from scipy.spatial.transform import Slerp

def polygon(rad, n, shift=False, zShift=0):
    theta = 2*np.pi/n
    if shift is True:
        #points = np.array([[0, 0, 0, 1]] + [ [rad*np.sin(theta*(i + 0.5)), rad*np.cos(theta*(i + 0.5)), 0, 1] for i in range(n)], dtype=np.float32)
        points = np.array([ [rad*np.sin(theta*(i + 0.5)), rad*np.cos(theta*(i + 0.5)), zShift, 1] for i in range(n)] , dtype=np.float32)
    else:
        points = np.array([ [rad*np.sin(theta*i), rad*np.cos(theta*i), zShift, 1] for i in range(n)], dtype=np.float32)

    return points

class DSPoseEstimator:
    def __init__(self, cameraMatrix, distCoeffs):
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs
        #self.featureModel = featureModel # take this as an argument?

        self.flag = cv.SOLVEPNP_ITERATIVE

        # pose relative to the pose at last update
        # updated when predict is called
        self.auvTranslation = np.array(3)
        self.auvRotation = np.array((3,3))
        self.auvTranslationCov = np.array(3)
        self.auvRotationCov = np.array((3,3))

        # relative pose of the docking station w.r.t the AUV
        # updated when update is called
        self.dsTranslation = np.array(3)
        self.dsTranslationCov = np.array(3)
        self.dsRotation = np.array((3,3))
        self.dsRotationCov = np.array((3,3))
        
        # estimated velocity of the docking station in the body frame
        # updated when update is called
        self.dsVelocity = np.array(6)
        self.dsVelocityCov = np.array(6)

    def _propagateAuv(self, translation, rotation, vel, dt, translationCov=np.zeros(3), rotationCov=np.zeros((3,3))):
        v = np.matmul(rotation, np.array(vel[:3])*dt)
        w = np.array(vel[3:])*dt
        
        translation[0] += v[0]
        translation[1] += v[1]
        translation[2] += v[2]

        r = R.from_matrix(rotation)
        rw = R.from_rotvec(w)
        r = r*rw # multiplying from the right makes the rotation in the local frame

        rotation = np.array(list(r.as_matrix()))

        return translation, translationCov, rotation, rotationCov

    def predict(self, auvVel, dt):
        """
        predict the relative pose between the AUV and the docking station
        The AUV and docking station pose is relative the AUV pose from the last update
        """
        self.self.dsVelocity
        pass

    def _transformFeaturePoseToDSPose(self):
        """
        Measurements are the pose of the feature model (the lights) w.r.t the camera.
        We need to transform the pose using the model of the docking station
        to estimate the pose/velocity of the docking station.
        """
        pass

    def _filterPose(self, translation, rotation):
        self._transformFeaturePoseToDSPose()
        return translation, rotation

    def _estimateDsState(self, translation, rotation, dt):
        
        vel = (translation - self.dsTranslation)*dt
        vel = np.concatenate([vel, np.zeros(3)])

        return translation, rotation, vel

    def update(self, featurePoints, associatedPoints, pointCovariance, dt):
        """
        Assume that predict have been called just before this
        featurePoints - points of the feature model [m]
        associatedPoints - detected and associated points in the image [m]
        pointCovariance - uncertainty of the detected points
        """

        #if self.flag == cv.SOLVEPNP_EPNP:
        #    points_2D = associatedPoints.reshape((associatedPoints.shape[0], 1, associatedPoints.shape[1]))

        featurePoints = np.array(list(featurePoints[:, :3]))
        success, rotationVector, translationVector = cv.solvePnP(featurePoints,
                                                                 associatedPoints,
                                                                 self.cameraMatrix,
                                                                 self.distCoeffs,
                                                                 useExtrinsicGuess=True,
                                                                 tvec=np.array([[0.], [0.], [1.]]),
                                                                 rvec=np.array([[0.], [0], [0.]]),
                                                                 flags=cv.SOLVEPNP_ITERATIVE)


        projectedPoints, jacobian = cv.projectPoints(featurePoints, rotationVector, translationVector, self.cameraMatrix, self.distCoeffs)
        
        # AUV homing and docking for remote operations
        # https://www.sciencedirect.com/science/article/pii/S0029801818301367
        #jacobian - 10 * 14
        #poseCov = np.matmul(jacobian.transpose(), np.linalg.inv(pointCovariance))
        #poseCov = np.matmul(poseCov, jacobian)
        #poseCov = np.linalg.inv(poseCov)

        translation, rotation = self._filterPose(translationVector[:, 0], rotationVector[:, 0])
        translation, rotation, vel = self._estimateDsState(translation, rotation, dt)
        
        self.dsTranslation = translation
        self.dsRotation = rotation
        self.dsVelocity = vel
        
        return self.dsTranslation, self.dsRotation

if __name__ =="__main__":
    pass