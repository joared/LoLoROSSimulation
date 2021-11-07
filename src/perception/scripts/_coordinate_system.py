from scipy.spatial.transform import Rotation as R
import numpy as np

R.as_matrix = R.as_dcm

class CoordinateSystem:
    # https://stackoverflow.com/questions/11140163/plotting-a-3d-cube-a-sphere-and-a-vector-in-matplotlib
    def __init__(self, translation=(0,0,0), euler=(0, 0, 0), scale=1):
        r = R.from_euler("XYZ", euler)
        self.rotation = r.as_matrix()
        self.initialRotation = r.as_matrix()
        self.translation = list(translation)
        self.initialTranslation = list(translation)
        self.scale = scale

    def reset(self):
        self.translation = self.initialTranslation.copy()
        self.rotation = self.initialRotation

    def setTransform(self, trans, rot):
        self.translation = [_ for _ in trans]
        self.rotation = rot #[_ for _ in rot] # this is weird, creates a list with arrays

    def transformedPoints(self, points, referenceTranslation=(0,0,0)):
        """
        Local to global
        """
        transformedPoints = []
        for p in points:
            transformedPoints.append(np.matmul(self.rotation, np.array(p)))

        newPoints = []
        for p in transformedPoints:
            newPoints.append( [p1+p2-p3 for p1,p2,p3 in zip(p, self.translation, referenceTranslation)] )

        return newPoints

    def transformedPointsInv(self, points, referenceTranslation=(0,0,0)):
        """
        Global to local, is this even right or even used?
        """
        points = np.array(points) - np.array(self.translation)
        newPoints = list(np.matmul(np.linalg.inv(np.array(self.rotation)), points.transpose()).transpose())
        #points = np.array(points) - np.array(self.translation)
        #points = np.matmul(np.linalg.inv(np.array(self.rotation)), np.array(points).transpose()).transpose()

        return list(newPoints)

class CoordinateSystemArtist:
    def __init__(self, coordinateSystem):
        self.cs = coordinateSystem

        self.xAxis = None
        self.yAxis = None
        self.zAxis = None

    def draw(self, ax, referenceTranslation=(0,0,0)):
        origin = np.array(self.cs.translation) - np.array(referenceTranslation)
        rotation = np.array(self.cs.rotation)
        ax.plot3D(*zip(*[origin, origin + self.cs.scale*rotation[:, 0]]), color="r", linewidth=2)
        ax.plot3D(*zip(*[origin, origin + self.cs.scale*rotation[:, 1]]), color="g", linewidth=2)
        ax.plot3D(*zip(*[origin, origin + self.cs.scale*rotation[:, 2]]), color="b", linewidth=2)

        return self.artists()

    def artists(self):
        return [self.xAxis,
                self.yAxis,
                self.zAxis]

    def init(self, ax):
        self.xAxis = ax.plot3D([], [], [], color="r", linewidth=2)[0]
        self.yAxis = ax.plot3D([], [], [], color="g", linewidth=2)[0]
        self.zAxis = ax.plot3D([], [], [], color="b", linewidth=2)[0]

        return self.artists()

    def update(self, show=True, referenceTranslation=(0,0,0)):
        if show:
            origin = np.array(self.cs.translation) - np.array(referenceTranslation)
            rotation = np.array(self.cs.rotation)
            self.xAxis.set_data_3d(*zip(*[origin, origin + rotation[:, 0]]))
            self.yAxis.set_data_3d(*zip(*[origin, origin + rotation[:, 1]]))
            self.zAxis.set_data_3d(*zip(*[origin, origin + rotation[:, 2]]))
        else:
            self.xAxis.set_data_3d([], [], [])
            self.yAxis.set_data_3d([], [], [])
            self.zAxis.set_data_3d([], [], [])

        return self.artists()
