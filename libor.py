# %%
import numpy as np
from lib.map import Trajectory, TangentPlane, loadSBET
from lib.model import Correspondence

# %% Load data
t,lla,rpy = loadSBET("/media/topo/Data/Data/ALS/Vallet/01_Trj/01_InitialTrajectories/NavGrade/SBET_MILF18_200HZ.out")
correspondences = np.loadtxt("/media/topo/Data/Data/ALS/Vallet/02_P2P/libor_correctBor/cor_outputs/LiDAR_p2p.txt", delimiter=',')

# %% Tangent plane and trajectory setup
latlon = np.array([46.5, 6.5])*np.pi/180
tangentPlane = TangentPlane(latlon[0], latlon[1])
R_e2enu = tangentPlane.R_ecef2enu

trajectory = Trajectory(t, lla, rpy, tangentPlane)

poses_i = trajectory.interpolate(correspondences[:, 0], customRPY=False)
poses_j = trajectory.interpolate(correspondences[:, 1], customRPY=False)
# %% Mount setup
mount = {
    'leverArm': np.array([-0.042, 0.183, -0.021]).reshape(3,1),
    'R_s2b': np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
    'bor': np.array([-0.2126 , 0.09961, 0.19137]).reshape(3,1)*np.pi/180
}
# %% Correspondences setup

corTest = Correspondence(correspondences[0], poses_i[0], poses_j[0], mount, R_e2enu)

# %%
theta = np.zeros((3,1))
corTest.compute_l_hat()
corTest.compute_Rb2m()
corTest.computeA()
corTest.computeB(theta)
corTest.compute_w(theta)

print("A:\n", corTest.A)
print("B:\n", corTest.B)
print("R_b2m_i:\n", corTest.R_b2m_i)
print("R_b2m_j:\n", corTest.R_b2m_j)
print("w:\n", corTest.w)
# %%
print("p_i:\n", corTest.p_i)
print("p_j:\n", corTest.p_j)
# %%
print("rpy i:\n", poses_i[0].rpy *180/np.pi)
print("rpy j:\n", poses_j[0].rpy *180/np.pi)
# %%

pose_test = trajectory.interpolate(correspondences[0, 0:2], customRPY=True)
# %%
for i in range(len(pose_test)):
    print(f"Pose {i} at time {pose_test[i].t}:")
    print("LLA:", pose_test[i].lla)
    print("XYZ:", pose_test[i].xyz)
    print("RPY (rad):", pose_test[i].rpy)
    print("RPY (deg):", pose_test[i].rpy * 180/np.pi)
    print()
# %%

# %%
