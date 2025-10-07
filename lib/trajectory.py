
import numpy as np
import pyproj

proj_ecef = pyproj.Proj(proj='geocent', ellps='WGS84', datum='WGS84')
proj_lla = pyproj.Proj(proj='latlong', ellps='WGS84', datum='WGS84')

lla2ecefTransformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4978")
ecef2llaTransformer = pyproj.Transformer.from_crs("EPSG:4978", "EPSG:4326")


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