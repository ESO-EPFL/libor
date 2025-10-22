import glob
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from cycler import cycler
import numpy as np

from lib.rotations import *

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
    def __init__(self, rawCor, trj, mount, R_e2enu, sigmas, initGuess=None):
        poses_i = trj.interpolate(rawCor[:, 0], customRPY=True)
        poses_j = trj.interpolate(rawCor[:, 1], customRPY=True)

        self.corSet = []
        for k in range(len(rawCor)):
            self.corSet.append(Correspondence(rawCor[k], poses_i[k], poses_j[k], mount, R_e2enu))
            self.corSet[k].compute_l_hat()
            self.corSet[k].compute_Rb2m()
            self.corSet[k].computeA()
            self.corSet[k].computeB(mount['initBor'])
            self.corSet[k].compute_w(mount['initBor'])

        self.n = len(self.corSet)
        self.sigmas = sigmas
        sigmas['rp'] = np.radians(sigmas['rp'])
        sigmas['y'] = np.radians(sigmas['y'])
        self.buildP()
        self.buildW()

        self.A = np.empty((3*len(self.corSet), 3), dtype=np.float32)
        self.B = np.empty((3*len(self.corSet), 12*len(self.corSet)), dtype=np.float32)
        self.w = np.empty((3*len(self.corSet), 1), dtype=np.float32)

        if initGuess is not None:
            self.theta = initGuess
        else:
            self.theta = np.zeros((3,1))

        print(f"Model initialized with {self.n} correspondences.")
        print(f"Initial boresight angles: {np.rad2deg(self.theta.flatten())} °")
        res = np.hstack([c.w for c in self.corSet])
        print(f"Initial mean residual: {np.mean(np.linalg.norm(res, axis=0)):.3f} m")

        self.initResiduals = res


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

    def plotResiduals(self, cfg):
        res = np.hstack([c.w for c in self.corSet])
        print(f"Mean residual: {np.mean(np.linalg.norm(res, axis=0)):.3f} m")
        print(f"Med residual: {np.median(np.linalg.norm(res, axis=0)):.3f} m")
        print(f"Max residual: {np.max(np.linalg.norm(res, axis=0)):.3f} m")
        plt.hist(np.linalg.norm(res, axis=0), bins=50)
        plt.hist(np.linalg.norm(self.initResiduals, axis=0), bins=50)
        plt.xlabel('Residual norm (m)')
        plt.ylabel('Count')
        plt.title('Histogram of correspondence residuals')
        plt.grid()
        plt.legend(['Final res.', 'Initial res.'])
        if 'logFolder' in cfg:
            plt.savefig(cfg['logFolder'] + cfg['prj_name'] + '_hist.svg', dpi=300)
        else:
            plt.show()

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

    def solve(self, max_iter=20, tol=1e-12, verbose=True):
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
                print("Matrix not pos-def, using np.linalg.solve")
                delta_theta = np.linalg.solve(self.N + 1e-12*np.eye(3), rhs)

            self.delta_theta = delta_theta
            self.theta = self.theta + delta_theta

            if np.linalg.norm(delta_theta) < tol:
                if verbose:
                    print("Converged.")
                break

            residual_term = (self.A @ delta_theta) + self.w   # 3n x 1
            v = self.recover_v(residual_term, M_fact)

            self.v = v
            self.S = S
            self.M_fact = M_fact

            if verbose:
                print(f"[iter {it+1}] Δθ = {delta_theta.flatten()*180/np.pi} [deg]")
                print(f"[iter {it+1}] θ = {self.theta.flatten()*180/np.pi} [deg]")

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

        J_obs = 0.0
        J_cond = 0.0

        for k in range(n):
            i12 = 12 * k
            i3  = 3 * k

            v_k = v_full[i12:i12+12, 0:1]        # (12,1)
            w_k = w_full[i3:i3+3, 0:1]          # (3,1)
            B_k = self.corSet[k].B              # (3,12)
            A_k = self.corSet[k].A              # (3,3)

            J_obs += float(v_k.T @ Pk @ v_k)

            r_k = (A_k @ delta) + (B_k @ v_k) + w_k 
            J_cond += float(r_k.T @ Wk @ r_k)
        sigma0_sq = (J_obs + J_cond) / r
        sigma0 = np.sqrt(sigma0_sq)

        Cov_theta = sigma0_sq * np.linalg.inv(self.N)
        std_theta = np.sqrt(np.abs(np.diag(Cov_theta)))  
        print("\n=== A-posteriori estimates ===")
        print(f"Cost cond. {J_cond:.2f}, obs.:  {J_obs:.2f}, ratio: {J_cond/J_obs:.2f}")
        print(f"Redundancy r = {r}")
        print(f"a-posteriori sigma0 = {sigma0:.3f} [unit weight]")
        with np.printoptions(precision=9, suppress=True):
            print("\nParameter covariance (deg^2):\n", Cov_theta * (180/np.pi)**2)
        print("Parameter std dev (deg):", np.degrees(std_theta))

        return sigma0, Cov_theta, std_theta

def corrLoader(cfg):
    pathList = glob.glob(cfg['p2p_folder'] + '/*.*')
    nPerFile = cfg['n'] // len(pathList)
    correspondences = np.vstack([np.loadtxt(pathList[i], delimiter=',')[np.random.choice(np.arange(len(np.loadtxt(pathList[i], delimiter=','))), nPerFile, replace=False)] for i in range(len(pathList))])
    print("Loaded", len(correspondences), "correspondences.")
    return correspondences

epfl_colors = [
    "#007480",  # Canard
    "#B51F1F",  # Groseille
    "#413D3A",  # Ardoise
    "#00A79F",  # Léman
    "#FF0000",  # Rouge
    "#CAC7C7",  # Perle
]
mpl.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['axes.prop_cycle'] = cycler(color=epfl_colors)
plt.rcParams.update({
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'grid.color': '#CCCCCC',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'axes.grid': True,
    'font.size': 12,
    'font.family':  ('cmr10', 'STIXGeneral'),
    'lines.linewidth': 0.75,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
})
np.set_printoptions(precision=5, suppress=True)
