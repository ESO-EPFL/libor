import numpy as np
import pyproj
from scipy.spatial.transform import Rotation as R, Slerp

from lib.rotations import *

proj_ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
proj_lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

lla2ecefTransformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4978")
ecef2llaTransformer = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:4326")

from lib.rotations import *

class TangentPlane:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

        self.xyz0 = np.array(lla2ecefTransformer.transform(lat, lon, 0, radians=True)).reshape(1,3)

        self.R_ecef2enu = T_enu_ned() @ R_ned2e(lat, lon).T

class Trajectory:
    def __init__(self, t, lla, rpy, tp):
        self.t = t

        self.lla = lla
        self.ecef = np.dstack(lla2ecefTransformer.transform(lla[:, 0], lla[:, 1], lla[:, 2],radians=True))[0]
        self.xyz = tp.R_ecef2enu @ (self.ecef - tp.xyz0).T


        R_nedi2b = []
        R_nedi2e = []

        for i in range(len(t)):
            R_nedi2b.append(R_ned2b(rpy[i,0], rpy[i,1], rpy[i,2]))
            R_nedi2e.append(R_ned2e(lla[i,0], lla[i,1]))


        self.R_nedi2b = R.from_matrix(np.array(R_nedi2b))
        self.R_nedi2e = R.from_matrix(np.array(R_nedi2e))

        self.slerp_nedi2b = Slerp(t, self.R_nedi2b)
        self.slerp_nedi2e = Slerp(t, self.R_nedi2e)
        
    def interpolate(self, timestamps):
        xyz_interp = np.empty((3, len(timestamps)))
        for i in range(3):
            xyz_interp[i,:] = np.interp(timestamps, self.t, self.xyz[i,:])
        
        R_ned2b_interp = self.slerp_nedi2b(timestamps)
        R_ned2e_interp = self.slerp_nedi2e(timestamps)

        return xyz_interp, R_ned2b_interp, R_ned2e_interp