from scipy.linalg import cho_factor, cho_solve
import numpy as np

from lib.rotations import *

import logging
logger = logging.getLogger("Libor")

def corrLoader(cfg):
    logger.info(f"Sampling strategy: {cfg['sampling']['strategy']}")

    correspondences = []

    if cfg['sampling']['strategy'] == 'freq':
        logger.info("Shuffling correspondences by regular time interval")
        logger.info(f"dt = {1/cfg['sampling']['value']:.3f} s")
        dt = 1/cfg['sampling']['value']
        
        for i in range(len(cfg["file_paths"])):
            corres_i = np.loadtxt(cfg["file_paths"][i], delimiter=',')
            corres_i = corres_i[np.argsort(corres_i[:,0])]
            t_int = [corres_i[0,0], corres_i[-1,0]]

            time = np.arange(t_int[0], t_int[1], dt)
            indexes = np.empty(len(time), dtype=int)
            for j in range(len(time)):
                indexes[j] = np.argmin(np.abs(corres_i[:,0] - time[j]))
            
            corres_i = corres_i[np.unique(indexes)]
            logger.info(f"Effective mean sampling rate for {cfg['file_paths'][i].split('/')[-1]}: {np.median(np.diff(corres_i[:,0])):.3f} s")   
            correspondences.append(corres_i)    
        correspondences = np.vstack(correspondences) 

    elif cfg['sampling']['strategy'] == 'time_window':
        logger.info("Shuffling correspondences by regular time window")
        logger.info(f"Time window = {cfg['sampling']['value']} s")
        
        for i in range(len(cfg["file_paths"])):
            corres_i = np.loadtxt(cfg["file_paths"][i], delimiter=',')
            corres_i = corres_i[np.argsort(corres_i[:,0])]
            mean_t = np.mean(corres_i[:,0])

            t_start = mean_t - cfg['sampling']['value']/2
            t_end = mean_t + cfg['sampling']['value']/2

            corres_i = corres_i[(corres_i[:,0] >= t_start) & (corres_i[:,0] <= t_end)]
            if corres_i.shape[0] > cfg['sampling']['max_per_file']:
                corres_i = corres_i[np.random.choice(np.arange(corres_i.shape[0]), cfg['sampling']['max_per_file'], replace=False)]
            logger.info(f"Effective time window for {cfg['file_paths'][i].split('/')[-1]}: {corres_i[0,0]:.3f} s to {corres_i[-1,0]:.3f} s")   
            correspondences.append(corres_i)    

        correspondences = np.vstack(correspondences)
            
    elif cfg['sampling']['strategy'] == 'max':
        nPerFile = cfg['sampling']['value']// len(cfg["file_paths"])
        logger.info("Shuffling correspondences by max count per file")
        logger.info(f"Keeping {nPerFile} correspondences per file")
        correspondences = np.vstack([np.loadtxt(cfg["file_paths"][i], delimiter=',')[np.random.choice(np.arange(len(np.loadtxt(cfg["file_paths"][i], delimiter=','))),
                                                                                            nPerFile,
                                                                                            replace=False)] for i in range(len(cfg["file_paths"]))])
    else:
        logger.info("No valid sampling strategy specified, loading all correspondences")
        correspondences = np.vstack([np.loadtxt(cfg["file_paths"][i], delimiter=',') for i in range(len(cfg["file_paths"]))])
    logger.info(f"Loaded {len(correspondences)} correspondences from {len(cfg['file_paths'])} files.")
    return correspondences

class Correspondence:   
    def __init__(self, corr, pose_i, pose_j, mount, R_e2enu):
        self.t_i = corr[0]
        self.t_j = corr[1]

        self.R_s2b = mount['R_s2b']
        self.leverArm = mount['leverArm']

        self.l = np.hstack((pose_i.xyz, pose_i.rpy, pose_j.xyz, pose_j.rpy)).reshape(12,1)
        self.l_hat = np.empty((12,1))

        self.u_i = self.R_s2b @ corr[2:5].reshape(3,1)
        self.u_j = self.R_s2b @ corr[5:8].reshape(3,1)

        self.U_i = skewT(self.u_i)
        self.U_j = skewT(self.u_j)        

        self.R_e2enu = R_e2enu

        self.R_ned2e_i = pose_i.R_ned2e
        self.R_ned2e_j = pose_j.R_ned2e

        self.A = np.zeros((3,3))
        self.B = np.zeros((3,12))
        self.v = np.zeros((12,1))
        self.w = np.zeros((3,1))
        self.p_i = np.zeros((3,1))
        self.p_j = np.zeros((3,1))
        self.P = np.zeros((12,12))

    def compute_l_hat(self):
        self.l_hat =  self.l + self.v

    def compute_Rb2m(self):
        r_i, p_i, y_i = self.l_hat[3:6].flatten()
        r_j, p_j, y_j = self.l_hat[9:12].flatten()

        R_b2ned_i = R_b2ned(r_i, p_i, y_i)
        R_b2ned_j = R_b2ned(r_j, p_j, y_j)

        self.R_b2m_i = self.R_e2enu @ self.R_ned2e_i @ R_b2ned_i
        self.R_b2m_j = self.R_e2enu @ self.R_ned2e_j @ R_b2ned_j

    def computeA(self):
        self.A = self.R_b2m_i @ self.U_i - self.R_b2m_j @ self.U_j
        assert self.A.shape == (3,3)

    def computeB(self, theta):

        r_i, p_i, y_i = self.l_hat[3:6].flatten()
        r_j, p_j, y_j = self.l_hat[9:12].flatten()

        s_i = self.u_i + self.U_i @ theta + self.leverArm
        s_j = self.u_j + self.U_j @ theta + self.leverArm

        dR_dr_i = dR_b2ned_dr(r_i, p_i, y_i)
        dR_dp_i = dR_b2ned_dp(r_i, p_i, y_i)
        dR_dy_i = dR_b2ned_dy(r_i, p_i, y_i)

        dR_dr_j = dR_b2ned_dr(r_j, p_j, y_j)
        dR_dp_j = dR_b2ned_dp(r_j, p_j, y_j)
        dR_dy_j = dR_b2ned_dy(r_j, p_j, y_j)

        B0 = np.eye(3)
        B1 = dR_dr_i @ s_i
        B2 = dR_dp_i @ s_i
        B3 = dR_dy_i @ s_i
        B4 = -np.eye(3)
        B5 = -dR_dr_j @ s_j
        B6 = -dR_dp_j @ s_j
        B7 = -dR_dy_j @ s_j

        self.B = self.R_e2enu @ np.hstack((B0, B1, B2, B3, B4, B5, B6, B7))
        assert self.B.shape == (3,12)

    def compute_w(self, theta):
        borVector_i = self.U_i @ theta
        borVector_j = self.U_j @ theta

        self.p_i = self.l_hat[0:3].reshape(3,1) + self.R_b2m_i @ (self.u_i + borVector_i + self.leverArm)
        self.p_j = self.l_hat[6:9].reshape(3,1) + self.R_b2m_j @ (self.u_j + borVector_j + self.leverArm)
        
        self.w = self.p_i - self.p_j
        assert self.w.shape == (3,1)

class Model:
    def __init__(self, rawCor, trj, cfg, R_e2enu):
        mount = cfg['mount']
        sigmas = cfg['sigmas']

        poses_i = trj.interpolate(rawCor[:, 0], customRPY=True)
        poses_j = trj.interpolate(rawCor[:, 1], customRPY=True)

        self.initBor = np.radians(mount['initBor'])
        self.refBor = np.radians(cfg['refBor'])

        self.theta = np.radians(mount['initBor'])

        self.corSet = []
        for k in range(len(rawCor)):
            self.corSet.append(Correspondence(rawCor[k], poses_i[k], poses_j[k], mount, R_e2enu))
            self.corSet[k].compute_l_hat()
            self.corSet[k].compute_Rb2m()
            self.corSet[k].computeA()
            self.corSet[k].computeB(self.initBor)
            self.corSet[k].compute_w(self.initBor)

        self.n = len(self.corSet)
        self.redundancy = 3*self.n - 3
        self.sigmas = sigmas
        self.sigmas['rp'] = np.radians(sigmas['rp'])
        self.sigmas['y'] = np.radians(sigmas['y'])
        self.buildP()
        self.buildW()

        self.A = np.empty((3*len(self.corSet), 3), dtype=np.float32)
        self.B = np.empty((3*len(self.corSet), 12*len(self.corSet)), dtype=np.float32)
        self.w = np.empty((3*len(self.corSet), 1), dtype=np.float32)
    
        self.initResiduals = np.linalg.norm(np.hstack([c.w for c in self.corSet]), axis=0)

        logger.info(f"Model initialized with {self.n} correspondences.")
        logger.info(f"Initial boresight angles: {np.rad2deg(self.theta.flatten())} °")
        logger.info(f"Initial mean residual: {np.mean(self.initResiduals):.3f} m")

    def buildP(self):
        """
        Build prior covariance matrix P, one block only (not full 12n x 12n block diagonal)
        """
        self.P_block = np.diag([
            1/self.sigmas['xy']**2,
            1/self.sigmas['xy']**2,
            1/self.sigmas['z']**2,
            1/self.sigmas['rp']**2,
            1/self.sigmas['rp']**2,
            1/self.sigmas['y']**2,
            1/self.sigmas['xy']**2,
            1/self.sigmas['xy']**2,
            1/self.sigmas['z']**2,
            1/self.sigmas['rp']**2,
            1/self.sigmas['rp']**2,
            1/self.sigmas['y']**2
        ]).astype(np.float32)

    def buildW(self):
        """
        Build observation weight matrix W, one block only (not full 3n x 3n block diagonal)
        """
        sigma = self.sigmas['p2p']
        self.W_block = np.diag([
            1/sigma**2,
            1/sigma**2,
            1/sigma**2
        ]).astype(np.float32)

    def stackBlocks(self):
        self.A[:,:] = np.vstack([c.A for c in self.corSet])
        self.w[:,:] = np.vstack([c.w for c in self.corSet])

        for k, c in enumerate(self.corSet):
            self.B[3*k:3*(k+1), 12*k:12*(k+1)] = c.B

    def compute_S(self):
        """
        Compute Schur complement:
        M = sum_k( B_k^T W_k B_k + P_k )   (block-diagonal)
        S = blockdiag( W_k - W_k B_k M_k^{-1} B_k^T W_k )
        Returns:
        S        : (3n x 3n) sparse block-diagonal Schur complement
        M_blocks : list of (M_k, M_fact_k)
        """
        from scipy.linalg import cho_factor, cho_solve
        import scipy.sparse as sp

        S_blocks = []
        M_blocks = []

        for c in self.corSet:
            Bk = c.B
            Wk = self.W_block
            Pk = self.P_block

            # Build M_k = Bk^T Wk Bk + Pk   (12x12)
            M_k = Bk.T @ Wk @ Bk + Pk

            # Factorize (Cholesky)
            M_fact_k = cho_factor(M_k, overwrite_a=False, check_finite=False)

            # Compute M_k^{-1} (B^T W)
            Xk = cho_solve(M_fact_k, Bk.T @ Wk, check_finite=False)

            # Compute S_k = Wk - Wk (Bk Xk)
            S_k = Wk - Wk @ (Bk @ Xk)

            S_blocks.append(S_k)
            M_blocks.append((M_k, M_fact_k))

        S = sp.block_diag(S_blocks, format='csr')

        return S, M_blocks
  
    def recover_v(self, residual_term, M_blocks):
        """
        Recover v from: v_k = - M_k^{-1} B_k^T W_k r_k  (blockwise)
        """
        v_list = []
        r = residual_term.reshape(-1, 3) 

        for k, c in enumerate(self.corSet):
            Bk = c.B
            Wk = self.W_block
            (_, M_fact_k) = M_blocks[k]

            rhs = Bk.T @ Wk @ r[k].reshape(3,1)
            v_k = -cho_solve(M_fact_k, rhs, check_finite=False)
            v_list.append(v_k)

        return np.vstack(v_list)

    def solve(self, max_iter=20, tol=1e-12):
        """
        Simple iterative Gauss-Helmert solver that uses compute_S and recover_v.
        Updates self.theta and returns (theta, v, info).
        """

        for it in range(max_iter):
            for c in self.corSet:
                c.compute_l_hat()
                c.compute_Rb2m()
                c.computeA()
                c.computeB(self.theta)
                c.compute_w(self.theta)

            self.stackBlocks()

            S, M_fact = self.compute_S()

            # reduced normal eqns: (A^T S A) delta = - A^T S w
            self.N = self.A.T @ S @ self.A                 # 3x3
            rhs = - self.A.T @ S @ self.w             # 3x1

            try:
                cfN = cho_factor(self.N, overwrite_a=False, check_finite=False)
                delta_theta = cho_solve(cfN, rhs, check_finite=False)
            except Exception:
                logger.warning("Matrix not pos-def, using np.linalg.solve")
                delta_theta = np.linalg.solve(self.N + 1e-12*np.eye(3), rhs)

            self.delta_theta = delta_theta
            self.theta = self.theta + delta_theta

            if np.linalg.norm(delta_theta) < tol:
                self.adjustedResiduals = np.linalg.norm(np.hstack([c.w for c in self.corSet]), axis=0)
                break

            residual_term = (self.A @ delta_theta) + self.w   # 3n x 1
            v = self.recover_v(residual_term, M_fact)

            self.v = v
            self.S = S
            self.M_fact = M_fact


            logger.info(f"[iter {it+1}] Δθ = {delta_theta.flatten()*180/np.pi} [deg]")
            logger.info(f"[iter {it+1}] θ = {self.theta.flatten()*180/np.pi} [deg]")

        return self.theta

    def computePosteriorUncertainty(self):
        """
        Blockwise computation of a-posteriori variance factor and parameter
        covariance (adapted to block-diagonal P and W).
        Returns:
            sigma0, Cov_theta, std_theta
        """
        n = self.n

        n_obs = 3 * n
        r = n_obs - 3
        if r <= 0:
            raise RuntimeError("Not enough redundancy (r <= 0)")

        v_full = self.v.reshape(-1, 1)
        w_full = self.w.reshape(-1, 1)
        delta = self.delta_theta.reshape(3, 1)

        Pk = self.P_block        
        Wk = self.W_block        

        self.J_obs = 0.0
        self.J_cond = 0.0

        for k in range(n):
            i12 = 12 * k
            i3  = 3 * k

            v_k = v_full[i12:i12+12, 0:1]        # (12,1)
            w_k = w_full[i3:i3+3, 0:1]          # (3,1)
            B_k = self.corSet[k].B              # (3,12)
            A_k = self.corSet[k].A              # (3,3)

            self.J_obs  += float((v_k.T @ Pk @ v_k).item())

            r_k = (A_k @ delta) + (B_k @ v_k) + w_k 
            self.J_cond += float((r_k.T @ Wk @ r_k).item())
        sigma0_sq = (self.J_obs + self.J_cond) / r
        sigma0 = np.sqrt(sigma0_sq)

        Cov_theta = sigma0_sq * np.linalg.inv(self.N)
        std_theta = np.sqrt(np.abs(np.diag(Cov_theta)))


        self.std_theta = std_theta
        self.Cov_theta = Cov_theta
        self.sigma0 = sigma0
    
    def marginalise(self, factor=5.0):
        """
        Marginalize (remove) outlier correspondences based on residual magnitude.
        factor : threshold multiplier (default 5x median)
        """
        res = np.hstack([c.w for c in self.corSet])
        res_norms = np.linalg.norm(res, axis=0)
        med_res = np.median(res_norms)
        self.thr = factor * med_res

        inliers = res_norms <= self.thr

        self.n_out_frac = [np.sum(~inliers) , self.n]

        self.corSet = [c for i, c in enumerate(self.corSet) if inliers[i]]
        self.n = len(self.corSet)

        self.A = np.empty((3*self.n, 3), dtype=np.float32)
        self.B = np.empty((3*self.n, 12*self.n), dtype=np.float32)
        self.w = np.empty((3*self.n, 1), dtype=np.float32)
        self.stackBlocks()

        self.theta_refined = self.solve(max_iter=1)      
        self.adjustedResiduals = np.linalg.norm(np.hstack([c.w for c in self.corSet]), axis=0)   
 
    def computeMapDensity(self, planimetric=True):
        """
        Characterise the spatial density of correspondences in the mapping frame.

        For each correspondence we take the mid-point of the matched pair (p_i, p_j)
        in the mapping frame and measure the Euclidean distance to its nearest
        neighbour among all other correspondences. This converts the temporal
        sampling rate into an interpretable on-map spacing.

        planimetric : if True, use horizontal (E-N) distance only -> footprint
                    density; if False, full 3D spacing.
        """
        from scipy.spatial import cKDTree

        # one representative map-frame point per correspondence
        pts = np.array([(0.5 * (c.p_i + c.p_j)).flatten() for c in self.corSet])
        if planimetric:
            pts = pts[:, :2]   # mapping frame is ENU -> keep East, North

        tree = cKDTree(pts)
        dists, _ = tree.query(pts, k=2)   # col 0 is self (d=0), col 1 is true NN
        nn = dists[:, 1]

        self.mapDensity = {
            "nn_mean":   float(np.mean(nn)),
            "nn_median": float(np.median(nn)),
            "nn_std":    float(np.std(nn)),
            "nn_min":    float(np.min(nn)),
            "nn_max":    float(np.max(nn)),
        }

        logger.info(
            f"Map-frame NN spacing ({'2D' if planimetric else '3D'}): "
            f"mean={self.mapDensity['nn_mean']:.2f} m, "
            f"median={self.mapDensity['nn_median']:.2f} m "
            f"(n={len(self.corSet)})"
        )
    
    def estimateObservability(self):

        if not hasattr(self, "N") or not hasattr(self, "Cov_theta"):
            raise RuntimeError("Run solve() and computePosteriorUncertainty() first.")

        eigvals, eigvecs = np.linalg.eigh(self.N)
        eigvals = np.sort(eigvals)[::-1]

        lambda_max = eigvals[0]
        lambda_min = eigvals[-1]
        cond_number = lambda_max / lambda_min

        D = np.sqrt(np.diag(self.Cov_theta))
        Corr = self.Cov_theta / np.outer(D, D)

        self.observability = {
            "eigvals": eigvals,
            "cond_number": cond_number,
            "corr_matrix": Corr,
            "eigvecs": eigvecs
        }

