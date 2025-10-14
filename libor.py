# %%
import numpy as np
from lib.map import Trajectory, TangentPlane, loadSBET
from lib.model import Model
import time

# %% General setup
t,lla,rpy = loadSBET("/media/topo/Data/Data/ALS/Vallet/01_Trj/01_InitialTrajectories/NavGrade/SBET_MILF18_200HZ.out")
cor_path = ["/media/topo/Data/Data/ALS/Vallet/02_P2P/libor_correctBor/line2-3/cor_outputs/LiDAR_p2p.txt",
            "/media/topo/Data/Data/ALS/Vallet/02_P2P/libor_correctBor/line2-4/cor_outputs/LiDAR_p2p.txt",
            "/media/topo/Data/Data/ALS/Vallet/02_P2P/libor_correctBor/line3-4/cor_outputs/LiDAR_p2p.txt"]

#randomly sample 500 corr per file and vstack them
n = 500
correspondences = np.vstack([np.loadtxt(cor_path[i], delimiter=',')[np.random.choice(np.arange(len(np.loadtxt(cor_path[i], delimiter=','))), n, replace=False)] for i in range(len(cor_path))])

t_span = [396525.0, 396965.0] 

refBor =  np.array([-0.2126 , 0.09961, 0.19137]).reshape(3,1)
mount = {
    'leverArm': np.array([-0.042, 0.183, -0.021]).reshape(3,1),
    'R_s2b': np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
    'bor': np.array([0.0 , 0.0, 0.0]).reshape(3,1)*np.pi/180
}

latlon = np.array([46.5, 6.5])*np.pi/180
tangentPlane = TangentPlane(latlon[0], latlon[1])
R_e2enu = tangentPlane.R_ecef2enu

trajectory = Trajectory(t, lla, rpy, tangentPlane, t_span)
sigmas = {
    'xy': 0.02,
    'z': 0.03,
    'rp': 0.02*np.pi/180,
    'y': 0.03*np.pi/180,
    'p2p': 0.065
}


# %%
t_0 = time.time()
model = Model(correspondences, trajectory, mount, R_e2enu, sigmas)
theta_hat, v_hat = model.solve(max_iter=5, tol=1e-6, verbose=True)

print("\n=== Solution ===")
print("Estimated boresight angles (rad):", theta_hat.flatten())
print("Estimated boresight angles (deg):", np.rad2deg(theta_hat.flatten()))
print("Difference from reference (deg):", (np.rad2deg(theta_hat) - refBor).flatten())
t_end = time.time()
print(f"\nTime elapsed: {t_end - t_0:.2f} seconds")
# --- 7. Diagnostics ---
model.plotResiduals()

# %%
