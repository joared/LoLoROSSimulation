import rospy
import numpy as np
from scipy.spatial.transform import Rotation as R, rotation

from tf.transformations import quaternion_multiply


if __name__ =="__main__":
    rospy.init_node("pose_estimation")