import time
import yaml
import argparse
import glob

import numpy as np
from lib.map import Trajectory, TangentPlane, loadSBET
from lib.model import Model

parser = argparse.ArgumentParser(description='Point cloud matching pipeline')
parser.add_argument('--yml','-y', type=str, help='Path to yml configuration file')
args = parser.parse_args()

cfg = yaml.safe_load(open(args.yml, 'r'))

t,lla,rpy = loadSBET(cfg['trj'])

cor_path = glob.glob(cfg['p2p_folder'] + '/*')

print("Loaded", len(cor_path), "correspondence files.")
nPerFile = int(cfg['n']/len(cor_path))

# t,lla,rpy = loadSBET("/media/topo/Data/Data/ALS/HelimapArpette/05_DN/02_ODyN_out/ArpetteL2L_all_SBET.out")
# cor_path = ["/media/topo/Data/Data/ALS/HelimapArpette/04_P2P/cor_outputs/icp/out082047_082434.csv",
#             "/media/topo/Data/Data/ALS/HelimapArpette/04_P2P/cor_outputs/icp/out082806_083149.csv",
#             "/media/topo/Data/Data/ALS/HelimapArpette/04_P2P/cor_outputs/icp/out085349_085810.csv",
#             "/media/topo/Data/Data/ALS/HelimapArpette/04_P2P/cor_outputs/icp/out090603_091008.csv",
#             "/media/topo/Data/Data/ALS/HelimapArpette/04_P2P/cor_outputs/icp/out092219_092622.csv",
#             "/media/topo/Data/Data/ALS/HelimapArpette/04_P2P/cor_outputs/icp/out093027_093422.csv",
#             "/media/topo/Data/Data/ALS/HelimapArpette/04_P2P/cor_outputs/icp/out093801_094107.csv"]
# refBor =  np.array([0.2061 , 0.02453, 0.35242]).reshape(3,1)
# mount = {
#     'leverArm': np.array([0.0, 0.0, 0.15]).reshape(3,1),
#     'R_s2b': np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]),
#     'bor': np.array([0.0 , 0.0, 0.0]).reshape(3,1)*np.pi/180
# }
# t_span = [288500.0, 294500.0] 
# latlon = np.array([46.0, 7.0])*np.pi/180
# n = 220

latlon = np.array(cfg['tp_latlon'])*np.pi/180
tangentPlane = TangentPlane(latlon[0], latlon[1])
R_e2enu = tangentPlane.R_ecef2enu

trajectory = Trajectory(t, lla, rpy, tangentPlane, cfg['t_span'])
sigmas = {
    'xy': 0.1,
    'z': 0.15,
    'rp': 0.04*np.pi/180,
    'y': 0.06*np.pi/180,
    'p2p': 0.10
}

correspondences = np.vstack([np.loadtxt(cor_path[i], delimiter=',')[np.random.choice(np.arange(len(np.loadtxt(cor_path[i], delimiter=','))), nPerFile, replace=False)] for i in range(len(cor_path))])

print("Total number of correspondences:", correspondences.shape)
# %%
t_0 = time.time()
model = Model(correspondences, trajectory, cfg['mount'], R_e2enu, sigmas)
theta_hat, v_hat = model.solve(max_iter=5, tol=1e-6, verbose=True)

print("\n=== Solution ===")
print("Estimated boresight angles (rad):", theta_hat.flatten())
print("Estimated boresight angles (deg):", np.rad2deg(theta_hat.flatten()))
print("Difference from reference (deg):", (np.rad2deg(theta_hat) - cfg['refBor']).flatten())
t_end = time.time()
print(f"\nTime elapsed: {t_end - t_0:.2f} seconds")
# --- 7. Diagnostics ---
model.plotResiduals()

# %%
