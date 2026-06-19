import os
import time
import yaml
import argparse
import glob


import numpy as np
from lib.map import Trajectory, TangentPlane, loadSBET
from lib.model import Model, corrLoader
from lib.stats import CalibrationStats

import logging

def setup_logger(cfg):
    out_folder = cfg["output"]["folder"]
    os.makedirs(out_folder, exist_ok=True)

    log_path = os.path.join(
        out_folder,
        f"{cfg['prj_name']}.log"
    )

    logger = logging.getLogger("Libor")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    if os.path.exists(log_path):
        os.remove(log_path)
        
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S"
    )

    fh = logging.FileHandler(log_path)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.info(f"Libor calibration for {cfg['prj_name']}")
    logger.info(f"Logging to {log_path}")

    return logger


def main():
    np.set_printoptions(precision=3)

    parser = argparse.ArgumentParser(description='Point cloud matching pipeline')
    parser.add_argument('--cfg','-c', type=str, help='Path to yml configuration file')
    args = parser.parse_args()
    
    cfg = yaml.safe_load(open(args.cfg, 'r'))

    logger = setup_logger(cfg)

    logger.info("")
    logger.info(f"=== Libor config : {cfg['prj_name']} ===")
    for key, value in cfg.items():
        if key == "mount":
            for mount_key, mount_value in value.items():
                logger.info(f"  {mount_key}: {mount_value}")
        else:   
            logger.info(f"{key}: {value}")

    logger.info("")
    logger.info("=== Setup ===")
    tangentPlane = TangentPlane(cfg['tp_latlon'][0], cfg['tp_latlon'][1])

    t,lla,rpy = loadSBET(cfg['trj'])
    trajectory = Trajectory(t, lla, rpy, tangentPlane, cfg['t_span'])

    cfg["file_paths"] = glob.glob(os.path.join(cfg["p2p_folder"], "*.*"))
    correspondences = corrLoader(cfg)

    logger.info("")
    logger.info("=== Model Initialization ===")
    t_0 = time.time()
    model = Model(correspondences, trajectory, cfg, tangentPlane.R_ecef2enu)

    model.computeMapDensity(planimetric=False)

    logger.info("=== Solving ... ===")

    theta_hat = model.solve(max_iter=5, tol=1e-9)

    logger.info("Converged.")
    logger.info(f"Final adjusted median residuals: {np.median(model.adjustedResiduals):.3f} m")

    logger.info("")
    logger.info("=== Solution ===")

    logger.info(f"Reference boresight: {np.array(cfg['refBor']).flatten()} °")
    logger.info(f"Estimated boresight: {np.rad2deg(theta_hat.flatten())} °")
    logger.info(f"Diff. from reference: {(np.rad2deg(theta_hat) - cfg['refBor']).flatten()} °")
    logger.info(f"Mean residual = {np.mean(model.adjustedResiduals):.3f} m")
    solving_time = time.time() - t_0
    logger.info(f"Time elapsed: {solving_time:.2f} seconds")

    logger.info("")
    logger.info(f"=== Marginalisation ===")

    model.marginalise(factor=3.0)

    logger.info(f"Mean residual = {np.mean(model.adjustedResiduals):.3f} m")
    logger.info(f"Threshold = {model.thr:.3f} m ({4.0}× median)")
    logger.info(f"Removing {model.n_out_frac[0]}/{model.n_out_frac[1]} correspondences")
    logger.info(f"Refined boresight: {np.rad2deg(model.theta.flatten())} °")
    logger.info(f"Final diff. from reference: {(np.rad2deg(model.theta) - cfg['refBor']).flatten()} °")

    logger.info("")
    logger.info("=== A-posteriori estimates ===")

    model.computePosteriorUncertainty() 

    logger.info(f"Cost cond. {model.J_cond:.2f}, obs.:  {model.J_obs:.2f}, ratio: {model.J_cond/model.J_obs:.2f}, redundancy: {model.redundancy}")
    logger.info(f"a-posteriori sigma0 = {model.sigma0:.3f} [unit weight]")
    with np.printoptions(precision=8, suppress=True):
        logger.info(f"Parameter covariance (deg^2):\n{model.Cov_theta * (180/np.pi)**2}")
    logger.info(f"Parameter std dev (deg): {np.round(np.degrees(model.std_theta), 4)}")


    logger.info("")
    logger.info("=== Observability analysis ===")

    model.estimateObservability()

    logger.info(f"Eigenvalues: {model.observability['eigvals']}")
    logger.info(f"Condition number: {model.observability['cond_number']:.2e}")
    logger.info(f"Correlation matrix: {np.round(model.observability['corr_matrix'], 4)}")

    stats = CalibrationStats(cfg)
    stats.collect_from_model(model, solving_time)
    stats.plot_residuals()
    stats.plot_boresight_difference()
    stats.plot_correlation_matrix()

    report_path = os.path.join(
        cfg["output"]["folder"],
        f"{cfg['prj_name']}_calibration_report.pdf"
    )

    stats.generate_pdf_report(report_path)

if __name__ == "__main__":
    main()