import numpy as np
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

c = np.cos
s = np.sin

#mapping frame m refers to local enu tangent plane with specified, fixed, origin.
def R1(r):
    """
    Rotation matrix around the x-axis, r in radians
    """
    return np.array([[1,    0,    0],
                     [0, c(r), s(r)],
                     [0,-s(r), c(r)]])

def R2(p):
    """
    Rotation matrix around the y-axis, p in radians
    """
    return np.array([[c(p), 0,-s(p)],
                     [   0, 1,    0],
                     [s(p), 0, c(p)]])

def R3(y):
    """
    Rotation matrix around the z-axis, y in radians
    """
    return np.array([[ c(y), s(y), 0],
                     [-s(y), c(y), 0],
                     [    0,    0, 1]])

def D1(r):
    """
    Derivative of rotation matrix around the x-axis, r in radians
    """
    return np.array([[0,     0,     0],
                     [0,-s(r), c(r)],
                     [0,-c(r),-s(r)]])

def D2(p):
    """
    Derivative of rotation matrix around the y-axis, p in radians
    """
    return np.array([[-s(p), 0,-c(p)],
                     [    0, 0,     0],
                     [ c(p), 0,-s(p)]])

def D3(y):
    """
    Derivative of rotation matrix around the z-axis, y in radians
    """
    return np.array([[-s(y), c(y), 0],
                     [-c(y),-s(y), 0],
                     [    0,    0, 0]])

def R_b2ned(r, p, y):
    """
    Rotation matrix from body to local NED frame given roll (r), pitch (p), yaw (y) in radians
    """
    return (R1(r) @ R2(p) @ R3(y)).T

def R_ned2b(r, p, y):
    """
    Rotation matrix from body to local NED frame given roll (r), pitch (p), yaw (y) in radians
    """
    return R1(r) @ R2(p) @ R3(y)

def dR_b2ned_dr(r, p, y):
    """
    Derivative of rotation matrix from body to local NED frame with respect to roll (r) in radians
    """
    return R3(y).T @ R2(p).T @ D1(r).T

def dR_b2ned_dp(r, p, y):
    """
    Derivative of rotation matrix from body to local NED frame with respect to pitch (p) in radians
    """
    return R3(y).T @ D2(p).T @ R1(r).T

def dR_b2ned_dy(r, p, y):
    """
    Derivative of rotation matrix from body to local NED frame with respect to yaw (y) in radians
    """
    return D3(y).T @ R2(p).T @ R1(r).T

def R_ned2e(lat,lon):
    """
    Rotation matrix from local level NED to ECEF frame.
    :param lat, lon: latitude and longitude in radians
    :return: rotation matrix
    """
    return np.array([[ -s(lat)*c(lon),-s(lon), -c(lat)*c(lon)],
                     [ -s(lat)*s(lon), c(lon), -c(lat)*s(lon)],
                     [         c(lat),      0,        -s(lat)]])

def T_enu_ned():
    """
    Rotation matrix from local level ENU to NED frame and vice versa
    """
    return np.array([[0, 1, 0],
                     [1, 0, 0],
                     [0, 0,-1]])

def R_b2m(lat, lon, r, p, y, R_e2m):
    """
    Rotation matrix from body to mapping enu frame 
    """
    R_b2ned = R_b2ned(r, p, y)

    R_ned2e = R_ned2e(lat, lon)

    return R_e2m @ R_ned2e @ R_b2ned

def skew(u):
    """
    Skew symmetric matrix of a vector
    """
    assert u.shape == (3,1) or u.shape == (3,)
    u = u.flatten()

    return np.array([[   0, -u[2],  u[1]],
                     [ u[2],     0, -u[0]],
                     [-u[1],  u[0],     0]])

def skewT(u):
    """
    Transpose of skew symmetric matrix of a vector
    """
    assert u.shape == (3,1) or u.shape == (3,)
    u = u.flatten()

    return np.array([[   0,  u[2], -u[1]],
                     [-u[2],     0,  u[0]],
                     [ u[1], -u[0],     0]])

def rpy_from_R_ned2b(R, as_degrees=False):
    """
    Extract roll, pitch, yaw from rotation matrix from ned to body, SO eq. 3.21
    """
    if abs(R[2,0]) != 1:
        r = np.arctan2(R[1,2], R[2,2])
        p = -np.arcsin(R[0,2])
        y = np.arctan2(R[0,1], R[0,0])
    #TODO: Add gimbal lock case support

    if as_degrees:
        return np.array([r, p, y])*180/np.pi
    else:
        return np.array([r, p, y])