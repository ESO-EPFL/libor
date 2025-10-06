# %% Import Libraries
import numpy as np
from lib.trajectory import loadSBET
from lib.map import Trajectory, TangentPlane
from lib.rotations import skewT

# %% Load data

t,lla,rpy = loadSBET("/media/topo/Data/Data/ALS/Vallet/01_Trj/01_InitialTrajectories/NavGrade/SBET_MILF18_200HZ.out")
lasvec = np.loadtxt("/media/topo/Data/Data/ALS/Vallet/03_CLOUDS/00_NavGrade/localTP/line2/sample.txt")
# %%
latlon = np.array([46.5, 6.5])*np.pi/180
tangentPlane = TangentPlane(latlon[0], latlon[1])

trajectory = Trajectory(t, lla, rpy, tangentPlane)

# %%
leverArm = np.array([-0.042, 0.183, -0.021]).reshape(3,1)
R_lidar2b = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
bor_rpy = np.array([-0.2126 , 0.09961, 0.19137]).reshape(3,1)*np.pi/180

# %%

xyz_interp, R_b2m_interp = trajectory.interpolate(lasvec[:,0])
# %%
P = np.empty((len(lasvec),3))
for i in range(len(lasvec)):
    u = R_lidar2b @ lasvec[i,-3:]
    U = skewT(u)

    borEffect = U @ bor_rpy
    P[i,:] = np.squeeze(xyz_interp[:,i].reshape(3,1) + R_b2m_interp[i].as_matrix() @ (u.reshape(3,1) + U @ bor_rpy + leverArm))

# %%
dif = P - lasvec[:,1:4]
# %%
max = np.max(np.linalg.norm(dif, axis=1))
# %%
