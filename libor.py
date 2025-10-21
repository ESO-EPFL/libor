import time
import yaml
import argparse
import glob

import numpy as np
from lib.map import Trajectory, TangentPlane, loadSBET
from lib.model import Model

def main():
    print("\n=== Setup ===")
    parser = argparse.ArgumentParser(description='Point cloud matching pipeline')
    parser.add_argument('--cfg','-c', type=str, help='Path to yml configuration file')
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open(args.cfg, 'r'))

    tangentPlane = TangentPlane(cfg['tp_latlon'][0], cfg['tp_latlon'][1])

    t,lla,rpy = loadSBET(cfg['trj'])
    trajectory = Trajectory(t, lla, rpy, tangentPlane, cfg['t_span'])

    cor_path = glob.glob(cfg['p2p_folder'] + '/*.*')
    nPerFile = cfg['n'] // len(cor_path)
    correspondences = np.vstack([np.loadtxt(cor_path[i], delimiter=',')[np.random.choice(np.arange(len(np.loadtxt(cor_path[i], delimiter=','))), nPerFile, replace=False)] for i in range(len(cor_path))])
    print("Loaded", len(cor_path), "correspondence files.")
    print("\n=== Model Initialization ===")
    t_0 = time.time()
    model = Model(correspondences, trajectory, cfg['mount'], tangentPlane.R_ecef2enu, cfg['sigmas'])
    theta_hat, _ = model.solve(max_iter=5, tol=1e-6, verbose=True)

    print("\n=== Solution ===")
    print("Estimated boresight:", np.rad2deg(theta_hat.flatten()), " °")
    print("Diff. from reference:", (np.rad2deg(theta_hat) - cfg['refBor']).flatten(), " °")

    print(f"\nTime elapsed: {time.time() - t_0:.2f} seconds")

    model.plotResiduals()

if __name__ == "__main__":
    main()