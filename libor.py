import sys
import time
import yaml
import argparse

import numpy as np
from lib.map import Trajectory, TangentPlane, loadSBET
from lib.model import Model, corrLoader

def main():
    np.set_printoptions(precision=3)
    parser = argparse.ArgumentParser(description='Point cloud matching pipeline')
    parser.add_argument('--cfg','-c', type=str, help='Path to yml configuration file')
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open(args.cfg, 'r'))

    print(f"Libor calibration for {cfg['prj_name']} ")

    if 'logFolder' in cfg:
        print(f"Logging to {cfg['logFolder']}/libor.log")
        logFile = open(cfg['logFolder'] + '/libor' + cfg['prj_name'] + '.log', 'a')
        sys.stdout = logFile
        print(f"\n === Libor report : {cfg['prj_name']} ===")
        for key, value in cfg.items():
            print((key, value))

    print("\n=== Setup ===")
    tangentPlane = TangentPlane(cfg['tp_latlon'][0], cfg['tp_latlon'][1])

    t,lla,rpy = loadSBET(cfg['trj'])
    trajectory = Trajectory(t, lla, rpy, tangentPlane, cfg['t_span'])

    correspondences = corrLoader(cfg)
    
    print("\n=== Model Initialization ===")
    t_0 = time.time()
    model = Model(correspondences, trajectory, cfg['mount'], tangentPlane.R_ecef2enu, cfg['sigmas'])
    theta_hat = model.solve(max_iter=5, tol=1e-9, verbose=True)

    print("\n=== Solution ===")
    print(f"Reference boresight: {np.array(cfg['refBor']).flatten()} °")
    print("Estimated boresight:", np.rad2deg(theta_hat.flatten()), "°")
    print("Diff. from reference:", (np.rad2deg(theta_hat) - cfg['refBor']).flatten(), "°")
    
    print(f"\nTime elapsed: {time.time() - t_0:.2f} seconds")
    model.computePosteriorUncertainty()

    theta_refined = model.marginalise(cfg, factor=5.0)
    print("Refined boresight:", np.rad2deg(theta_refined.flatten()), " °")
    print("Final diff. from reference:", (np.rad2deg(theta_refined) - cfg['refBor']).flatten(), " °")
    model.computePosteriorUncertainty()

    model.plotResiduals(cfg)
    model.plotBorDiff(cfg)


if __name__ == "__main__":
    main()