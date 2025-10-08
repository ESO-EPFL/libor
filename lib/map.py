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

        self.lla = lla.reshape(-1,3)
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
        
    def interpolate(self, timestamps, customRPY = True):
        xyz_interp = np.empty((3, len(timestamps)))
        lla_interp = np.empty((len(timestamps),3))
        for i in range(3):
            xyz_interp[i,:] = np.interp(timestamps, self.t, self.xyz[i,:])
            lla_interp[i,:] = np.interp(timestamps, self.t, self.lla[i,:])

        R_nedi2b_interp = self.slerp_nedi2b(timestamps)
        R_nedi2e_interp = self.slerp_nedi2e(timestamps).as_matrix()

        if customRPY:
            rpy_interp = np.empty((len(timestamps),3))
            for i in range(len(timestamps)):
                rpy_interp[i,:] = rpy_from_R_ned2b(R_nedi2b_interp[i].as_matrix(), as_degrees=True)
        else:
            rpy_interp = R_nedi2b_interp.as_euler('xyz', degrees=True)


        return xyz_interp, R_nedi2b_interp, R_nedi2e_interp, rpy_interp
    
def loadSBET(path):
    """
    Decodes an APPLANIX SNV/SBET file.

    Parameters:
    - settings: path to SBET

    Returns:
    - data: numpy array of processed data

  Input record: 17xdouble=(136 bytes)
       0  time  			sec_of_week 
       1  latitude   		rad
       2  longitude  		rad
       3  altitude       meters
       4  x_wander_vel   m/s
       5  y_wander_vel   m/s
       6  z_wander_vel  	m/s
       7  roll          	radians
       8  pitch         	radians
       9  wander_heading radians
       10 wander angle   radians
       11 x body accel   m/s^2
       12 y body accel   m/s^2
       13 z body accel   m/s^2
       14 x angular rate rad/s
       15 y angular rate rad/s
	   16 z angular rate rad/s					
 This is what is written in the ouput record:
       0   time            sec_of_week
       1   latitude        rad
       2   longitude       rad
       3   altitude        m
       4   roll            rad
       5   pitch           rad
       6  heading         rad 
    """

    try:
        with open(path, "rb") as f:
            print(f"Loading file {path}")
            data = np.fromfile(f, dtype=np.float64).reshape(-1,17)
    except Exception as e:
        errmsg = f"Cannot open file! {str(e)}"
        raise ValueError(errmsg)
        
    return data[:, 0], data[:, 1:4],  np.column_stack((data[:, 7:9], data[:, 9]-data[:, 10]))