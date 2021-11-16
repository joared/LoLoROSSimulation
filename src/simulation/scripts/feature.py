import numpy as np
from scipy.spatial.transform import Rotation as R
from coordinate_system import CoordinateSystem, CoordinateSystemArtist

def polygon(rad, n, shift=False, zShift=0):
    theta = 2*np.pi/n
    if shift is True:
        #points = np.array([[0, 0, 0, 1]] + [ [rad*np.sin(theta*(i + 0.5)), rad*np.cos(theta*(i + 0.5)), 0, 1] for i in range(n)], dtype=np.float32)
        points = np.array([ [rad*np.sin(theta*(i + 0.5)), rad*np.cos(theta*(i + 0.5)), zShift, 1] for i in range(n)] , dtype=np.float32)
    else:
        points = np.array([ [rad*np.sin(theta*i), rad*np.cos(theta*i), zShift, 1] for i in range(n)], dtype=np.float32)

    return points

class FeatureModel(CoordinateSystem):
    def __init__(self, rads, ns, shifts, zShifts, euler=(0, 0, 0), *args, **kwargs):
        CoordinateSystem.__init__(self, *args, **kwargs)
        
        assert len(rads) == len(ns) == len(shifts) == len(zShifts)
        self.features = None
        for r, n, s, z in zip(rads, ns, shifts, zShifts):
            if self.features is None:
                self.features = polygon(r, n, s, z)
            else:
                self.features = np.append(self.features, polygon(r, n, s, z), axis=0)

        rotMat = R.from_euler("XYZ", euler).as_dcm()
        self.features = np.matmul(rotMat, self.features[:, :3].transpose()).transpose()
        self.features = self.features[:, :3].copy() # Don't need homogenious

class _FeatureModel(CoordinateSystem):
    def __init__(self, rad, n, shift=False, centerPointDist=None, zShift=0, *args, **kwargs):
        CoordinateSystem.__init__(self, *args, **kwargs)
        self.features = list(polygon(rad, n, shift, zShift)[:, :3]) # homogenious not needed
        if centerPointDist is not None:
            self.features.insert(0, np.array([0, 0, centerPointDist]))
        self.features = list(self.features)

class FeatureModelArtist3D(CoordinateSystemArtist):
    def __init__(self, feature, *args, **kwargs):
        CoordinateSystemArtist.__init__(self, feature, *args, **kwargs)
        self.feature = feature
        
        self.features3D = None
        #self.target3D = None    
        #self.featuresProj3D = None
        #self.referenceLines = []

    def artists(self):
        return [self.features3D] + CoordinateSystemArtist.artists(self)

    def init(self, ax):
        CoordinateSystemArtist.init(self, ax)
        self.features3D = ax.plot3D([], [], [], marker="o", color="m")[0]
        #self.features3D = ax.scatter([], [], [], color="navy")

        return self.artists()

    def update(self, showAxis=True, referenceTranslation=(0,0,0)):
        CoordinateSystemArtist.update(self, showAxis, referenceTranslation)
        points = self.feature.transformedPoints(self.feature.features, referenceTranslation)
        self.features3D.set_data_3d(*zip(*points + [points[0]])) # line
        #self.features3D._offsets3d = [*zip(*points + [points[0]])] # scatter

        return self.artists()

        
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    fm = FeatureModel([0.06, 0], [4, 1], [False, False], [0, 0.043])
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(*zip(*fm.features))
    size = 0.1
    ax.set_xlim(-size, size)
    ax.set_ylim(-size, size)
    ax.set_zlim(-size, size)
    plt.show()