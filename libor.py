# %%
import numpy as np
from lib.map import Trajectory, TangentPlane, loadSBET
from lib.model import Model
from matplotlib import pyplot as plt

# %% General setup
t,lla,rpy = loadSBET("/media/topo/Data/Data/ALS/Vallet/01_Trj/01_InitialTrajectories/NavGrade/SBET_MILF18_200HZ.out")
correspondences = np.loadtxt("/media/topo/Data/Data/ALS/Vallet/02_P2P/libor_correctBor/cor_outputs/LiDAR_p2p.txt", delimiter=',')
t_span = [396525.0, 396845.0] 

mount = {
    'leverArm': np.array([-0.042, 0.183, -0.021]).reshape(3,1),
    'R_s2b': np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
    #'bor': np.array([0.0 , 0.0, 0.0]).reshape(3,1)*np.pi/180
    'bor': np.array([-0.2126 , 0.09961, 0.19137]).reshape(3,1)*np.pi/180
}

latlon = np.array([46.5, 6.5])*np.pi/180
tangentPlane = TangentPlane(latlon[0], latlon[1])
R_e2enu = tangentPlane.R_ecef2enu

trajectory = Trajectory(t, lla, rpy, tangentPlane, t_span)
sigmas = {
    'xy': 0.02,
    'z': 0.02,
    'rp': 0.1*np.pi/180,
    'y': 0.2*np.pi/180,
    'p2p': 0.075
}


#%% Constraint instantiation
correspondences = correspondences[::100]  

model = Model(correspondences, trajectory, mount, R_e2enu, sigmas)

model.buildP()
model.plotP()
model.plotResiduals()


# %%
