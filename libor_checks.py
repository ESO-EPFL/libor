# %% Import Libraries
import numpy as np
from lib.map import Trajectory, TangentPlane, loadSBET
from lib.rotations import rpy_from_R_ned2b, skewT

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
xyz_interp, R_ned2b_interp, R_ned2e_interp, rpy_interp2 = trajectory.interpolate(lasvec[:,0])

rpy_interp = R_ned2b_interp.as_euler('xyz', degrees=True)
rpy_test = np.zeros((len(lasvec),3))
for i in range(len(lasvec)):
    rpy_test[i,:] = rpy_from_R_ned2b(R_ned2b_interp[i].as_matrix())


# %%
P = np.empty((len(lasvec),3))
for i in range(len(lasvec)):
    u = R_lidar2b @ lasvec[i,-3:]
    U = skewT(u)

    R_b2m = R_e2enu @ R_ned2e_interp[i].as_matrix() @ R_ned2b_interp[i].as_matrix().T

    borEffect = U @ bor_rpy
    P[i,:] = np.squeeze(xyz_interp[:,i].reshape(3,1) + R_b2m @ (u.reshape(3,1) + U @ bor_rpy + leverArm))

# %%
dif = P - lasvec[:,1:4]
# %%
max = np.max(np.linalg.norm(dif, axis=1))
# %% georef correspondences and check that differences match the expected one
xyz_interp_a, R_ned2b_interp_a, R_ned2e_interp_a, rpy_a_interp = trajectory.interpolate(corres[:,1])
xyz_interp_b, R_ned2b_interp_b, R_ned2e_interp_b, rpy_b_interp = trajectory.interpolate(corres[:,0])

Pa = np.empty((len(corres),3))
Pb = np.empty((len(corres),3))
for i in range(len(corres)):
    u_b = R_lidar2b @ corres[i,-6:-3]
    u_a = R_lidar2b @ corres[i,-3:]
    U_b = skewT(u_b)
    U_a = skewT(u_a)

    borEffect_b = U_b @ bor_rpy
    borEffect_a = U_a @ bor_rpy

    R_b2m_interp_b = R_e2enu @ R_ned2e_interp_b[i].as_matrix() @ R_ned2b_interp_b[i].as_matrix().T
    R_b2m_interp_a = R_e2enu @ R_ned2e_interp_a[i].as_matrix() @ R_ned2b_interp_a[i].as_matrix().T

    Pb[i,:] = np.squeeze(xyz_interp_b[:,i].reshape(3,1) + R_b2m_interp_b @ (u_b.reshape(3,1) + U_b @ bor_rpy + leverArm))
    Pa[i,:] = np.squeeze(xyz_interp_a[:,i].reshape(3,1) + R_b2m_interp_a @ (u_a.reshape(3,1) + U_a @ bor_rpy + leverArm))



# %%
dif_noRef = np.linalg.norm(Pa - Pb, axis=1)
print(f"Max difference before ICP: {np.max(dif_noRef)} m")
print(f"Mean difference before ICP: {np.mean(dif_noRef)} m")
print(f"Median difference before ICP: {np.median(dif_noRef)} m")
print(f"Std difference before ICP: {np.std(dif_noRef)} m")
# %% do same with icp
xyz_interp_a, R_ned2b_interp_a, R_ned2e_interp_a, rpy_a_interp = trajectory.interpolate(corres_icp[:,1])
xyz_interp_b, R_ned2b_interp_b, R_ned2e_interp_b, rpy_b_interp = trajectory.interpolate(corres_icp[:,0])

Pa = np.empty((len(corres_icp),3))
Pb = np.empty((len(corres_icp),3))
for i in range(len(corres_icp)):
    u_b = R_lidar2b @ corres_icp[i,-6:-3]
    u_a = R_lidar2b @ corres_icp[i,-3:]
    U_b = skewT(u_b)
    U_a = skewT(u_a)

    borEffect_b = U_b @ bor_rpy
    borEffect_a = U_a @ bor_rpy

    R_b2m_interp_b = R_e2enu @ R_ned2e_interp_b[i].as_matrix() @ R_ned2b_interp_b[i].as_matrix().T
    R_b2m_interp_a = R_e2enu @ R_ned2e_interp_a[i].as_matrix() @ R_ned2b_interp_a[i].as_matrix().T

    Pb[i,:] = np.squeeze(xyz_interp_b[:,i].reshape(3,1) + R_b2m_interp_b @ (u_b.reshape(3,1) + U_b @ bor_rpy + leverArm))
    Pa[i,:] = np.squeeze(xyz_interp_a[:,i].reshape(3,1) + R_b2m_interp_a @ (u_a.reshape(3,1) + U_a @ bor_rpy + leverArm))

rpy_interp_a = R_ned2b_interp_a.as_euler('xyz', degrees=True)
rpy_interp_b = R_ned2b_interp_b.as_euler('xyz', degrees=True)

rpy_test_a = np.zeros((len(corres_icp),3))
rpy_test_b = np.zeros((len(corres_icp),3))
for i in range(len(corres_icp)):
    rpy_test_a[i,:] = rpy_from_R_ned2b(R_ned2b_interp_a[i].as_matrix(), as_degrees=True)
    rpy_test_b[i,:] = rpy_from_R_ned2b(R_ned2b_interp_b[i].as_matrix(), as_degrees=True)
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
print("RPY differences a:")
print(f"Roll: mean {np.mean(rpy_interp_a[:,0]-rpy_test_a[:,0])} deg, std {np.std(rpy_interp_a[:,0]-rpy_test_a[:,0])} deg")
print(f"Pitch: mean {np.mean(rpy_interp_a[:,1]-rpy_test_a[:,1])} deg, std {np.std(rpy_interp_a[:,1]-rpy_test_a[:,1])} deg")
print(f"Yaw: mean {np.mean(rpy_interp_a[:,2]-rpy_test_a[:,2])} deg, std {np.std(rpy_interp_a[:,2]-rpy_test_a[:,2])} deg")
print("RPY differences b:")
print(f"Roll: mean {np.mean(rpy_interp_b[:,0]-rpy_test_b[:,0])} deg, std {np.std(rpy_interp_b[:,0]-rpy_test_b[:,0])} deg")
print(f"Pitch: mean {np.mean(rpy_interp_b[:,1]-rpy_test_b[:,1])} deg, std {np.std(rpy_interp_b[:,1]-rpy_test_b[:,1])} deg")
print(f"Yaw: mean {np.mean(rpy_interp_b[:,2]-rpy_test_b[:,2])} deg, std {np.std(rpy_interp_b[:,2]-rpy_test_b[:,2])} deg")
# %%