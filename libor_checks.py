# %% Import Libraries
import numpy as np
from lib.map import Trajectory, TangentPlane, loadSBET
from lib.rotations import R_ned2b, rpy_from_R_ned2b, skewT

# %% Load data

t,lla,rpy = loadSBET("/media/topo/Data/Data/ALS/Vallet/01_Trj/01_InitialTrajectories/NavGrade/SBET_MILF18_200HZ.out")
lasvec = np.loadtxt("/media/topo/Data/Data/ALS/Vallet/03_CLOUDS/00_NavGrade/localTP/line2/sample.txt")
corres = np.loadtxt("/media/topo/Data/Data/ALS/Vallet/02_P2P/libor_correctBor/cor_outputs/LiDAR_p2p_noRefinement.txt", delimiter=',')
corres_icp = np.loadtxt("/media/topo/Data/Data/ALS/Vallet/02_P2P/libor_correctBor/cor_outputs/LiDAR_p2p.txt", delimiter=',')
# %%
latlon = np.array([46.5, 6.5])*np.pi/180
tangentPlane = TangentPlane(latlon[0], latlon[1])
R_e2enu = tangentPlane.R_ecef2enu

trajectory = Trajectory(t, lla, rpy, tangentPlane)
# %%
leverArm = np.array([-0.042, 0.183, -0.021]).reshape(3,1)
R_lidar2b = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
bor_rpy = np.array([-0.2126 , 0.09961, 0.19137]).reshape(3,1)*np.pi/180

# %%
poses = trajectory.interpolate(lasvec[:,0])

R_ned2b_interp = []
R_ned2e_interp = []

for i in range(len(lasvec)):
    R_ned2b_interp.append(R_ned2b(poses[i].rpy[0], poses[i].rpy[1], poses[i].rpy[2]))
    R_ned2e_interp.append(poses[i].R_ned2e)

# %%
P = np.empty((len(lasvec),3))
for i in range(len(lasvec)):
    u = R_lidar2b @ lasvec[i,-3:]
    U = skewT(u)

    R_b2m = R_e2enu @ R_ned2e_interp[i] @ R_ned2b_interp[i].T

    borEffect = U @ bor_rpy
    P[i,:] = np.squeeze(poses[i].xyz.reshape(3,1) + R_b2m @ (u.reshape(3,1) + U @ bor_rpy + leverArm))

# %%
dif = P - lasvec[:,1:4]
# %%
max = np.max(np.linalg.norm(dif, axis=1))

print(f"Max difference before ICP: {max} m")    
#
# %% georef correspondences and check that differences match the expected one
poses_a = trajectory.interpolate(corres[:,1])
poses_b = trajectory.interpolate(corres[:,0])

Pa = np.empty((len(corres),3))
Pb = np.empty((len(corres),3))
for i in range(len(corres)):
    u_b = R_lidar2b @ corres[i,-6:-3]
    u_a = R_lidar2b @ corres[i,-3:]
    U_b = skewT(u_b)
    U_a = skewT(u_a)

    borEffect_b = U_b @ bor_rpy
    borEffect_a = U_a @ bor_rpy

    R_b2m_interp_b = R_e2enu @ poses_b[i].R_ned2e @ R_ned2b(poses_b[i].rpy[0], poses_b[i].rpy[1], poses_b[i].rpy[2]).T
    R_b2m_interp_a = R_e2enu @ poses_a[i].R_ned2e @ R_ned2b(poses_a[i].rpy[0], poses_a[i].rpy[1], poses_a[i].rpy[2]).T

    Pb[i,:] = np.squeeze(poses_b[i].xyz.reshape(3,1) + R_b2m_interp_b @ (u_b.reshape(3,1) + U_b @ bor_rpy + leverArm))
    Pa[i,:] = np.squeeze(poses_a[i].xyz.reshape(3,1) + R_b2m_interp_a @ (u_a.reshape(3,1) + U_a @ bor_rpy + leverArm))

# %%
dif_noRef = np.linalg.norm(Pa - Pb, axis=1)
print(f"Max difference before ICP: {np.max(dif_noRef)} m")
print(f"Mean difference before ICP: {np.mean(dif_noRef)} m")
print(f"Median difference before ICP: {np.median(dif_noRef)} m")
print(f"Std difference before ICP: {np.std(dif_noRef)} m")
# %% do same with icp
poses_a = trajectory.interpolate(corres_icp[:,1])
poses_b = trajectory.interpolate(corres_icp[:,0])

Pa = np.empty((len(corres_icp),3))
Pb = np.empty((len(corres_icp),3))
for i in range(len(corres_icp)):
    u_b = R_lidar2b @ corres_icp[i,-6:-3]
    u_a = R_lidar2b @ corres_icp[i,-3:]
    U_b = skewT(u_b)
    U_a = skewT(u_a)

    borEffect_b = U_b @ bor_rpy
    borEffect_a = U_a @ bor_rpy

    R_b2m_interp_b = R_e2enu @ poses_b[i].R_ned2e @ R_ned2b(poses_b[i].rpy[0], poses_b[i].rpy[1], poses_b[i].rpy[2]).T
    R_b2m_interp_a = R_e2enu @ poses_a[i].R_ned2e @ R_ned2b(poses_a[i].rpy[0], poses_a[i].rpy[1], poses_a[i].rpy[2]).T

    Pb[i,:] = np.squeeze(poses_b[i].xyz.reshape(3,1) + R_b2m_interp_b @ (u_b.reshape(3,1) + U_b @ bor_rpy + leverArm))
    Pa[i,:] = np.squeeze(poses_a[i].xyz.reshape(3,1) + R_b2m_interp_a @ (u_a.reshape(3,1) + U_a @ bor_rpy + leverArm))

# %%
dif = np.linalg.norm(Pa - Pb, axis=1)
print(f"Max difference after ICP: {np.max(dif)} m")
print(f"Mean difference after ICP: {np.mean(dif)} m")
print(f"Median difference after ICP: {np.median(dif)} m")
print(f"Std difference after ICP: {np.std(dif)} m")
# %%
import matplotlib.pyplot as plt
plt.hist(dif, bins=100)
plt.hist(dif_noRef, bins=100)
plt.xlabel('Distance (m)')
plt.ylabel('Count')
plt.title('Histogram of distances between corresponding points after ICP')
plt.grid(True)
plt.show()
# %%