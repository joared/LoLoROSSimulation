import numpy as np

from feature import FeatureModel
from feature_extraction import FeatureExtractor
from pose_estimation import PoseEstimator
from control import IBVSController, PBVSController


class VisualServo:
    def __init__(self, 
                 featureModel,
                 featureExtraction,
                 control):
                 
        self.featureModel = featureModel
        self.featureExtraction = featureExtraction
        self.control = control

    def callback(self):
        # projected image features
        detectedFeatures = self.featureExtractor.extract(self.featureModel, self.poseEstimator.current)
        pose = self.poseEstimator.estimate(detectedFeatures)
        vel, err = self.controller.control(pose)
        return err

    def gen(self):
        while True:
            err = self.callback()
            if np.linalg.norm(err) < 0.1:
                return
            yield err

if __name__ == "__main__":
    pbvs = True
    featureModel = FeatureModel()
    featureExtractor = FeatureExtractor()
    poseEstimator = PoseEstimator()
    controller = PBVSController() if pbvs else IBVSController()

    vs = VisualServo(featureModel, featureExtractor, poseEstimator, controller)