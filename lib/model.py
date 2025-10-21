import numpy as n
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from cycler import cycler

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

        self.A = np.empty((3*len(self.corSet), 3))
        self.B = np.empty((3*len(self.corSet), 12*len(self.corSet)))
        self.w = np.empty((3*len(self.corSet), 1))

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
        Pk = np.diag([
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
            ])


        self.P = np.zeros((12*self.n, 12*self.n))
        for k in range(self.n):
            self.P[12*k:12*(k+1), 12*k:12*(k+1)] = Pk

    def buildW(self):
        sigma = self.sigmas['p2p'] / 3
        Qxx_inv = np.diag([
            1/sigma**2,
            1/sigma**2,
            1/sigma**2
        ])

        self.W = np.zeros((3*self.n, 3*self.n))
        for k in range(self.n):
            self.W[3*k:3*(k+1), 3*k:3*(k+1)] = Qxx_inv

    def plotResiduals(self):
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
        plt.show()

    def stackBlocks(self):
        self.A[:,:] = np.vstack([c.A for c in self.corSet])
        self.w[:,:] = np.vstack([c.w for c in self.corSet])

        for k, c in enumerate(self.corSet):
            self.B[3*k:3*(k+1), 12*k:12*(k+1)] = c.B

    def compute_S(self):
        """
        Compute Schur complement:
          M = B^T W B + P      (12n x 12n)
          S = W - W B M^{-1} B^T W   (3n x 3n)

        Returns:
          S        : (3n x 3n) Schur-complement (symmetric)
          M_fact   : Cholesky factorization object for M (use with cho_solve)
        """
        B = self.B                    # (3n x 12n)
        W = self.W                    # (3n x 3n)
        P = self.P                    # (12n x 12n)

        # Build M = B^T W B + P
        # compute BtW = B.T @ W  -> (12n x 3n)
        BtW = B.T @ W
        M = BtW @ B + P              # (12n x 12n)

        # Factorize M (Cholesky), prefer lower-triangular (cho_factor returns (c, lower_flag))
        M_fact = cho_factor(M, overwrite_a=False, check_finite=False)

        # Solve M X = B^T W  for X  (i.e. X = M^{-1} (B^T W) ), using factor
        # X will be (12n x 3n)
        X = cho_solve(M_fact, BtW, check_finite=False)

        # Compute B @ X  -> (3n x 3n) equals B M^{-1} B^T W
        BXM = B @ X

        # Now S = W - W @ (B @ X)
        S = W - (W @ BXM)

        return S, M_fact
    
    def recover_v(self, residual_term, M_fact):
        """
        Recover v from:
        v = - M^{-1} B^T W (residual_term)
        where residual_term = A * delta_theta + w  (3n x 1)

        Inputs:
        residual_term : (3n x 1) array
        M_fact        : factorization returned by compute_S()
        Returns:
        v : (12n x 1)
        """
        # BtW = B.T @ W  (12n x 3n)  -- compute once
        BtW = self.B.T @ self.W

        # rhs = BtW @ residual_term  (12n x 1)
        rhs = BtW @ residual_term

        # solve M v = - rhs  (v = - M^{-1} rhs)
        v = -cho_solve(M_fact, rhs, check_finite=False)

        return v
    
    def solve(self, max_iter=20, tol=1e-12, verbose=True):
        """
        Simple iterative Gauss-Helmert solver that uses compute_S and recover_v.
        Updates self.theta and returns (theta, v, info).
        """

        for it in range(max_iter):
            # update per-correspondence (ensure all c.* depend on current theta)
            for c in self.corSet:
                c.compute_l_hat()
                c.compute_Rb2m()
                c.computeA()
                c.computeB(self.theta)
                c.compute_w(self.theta)

            # stack into big matrices
            self.stackBlocks()

            # compute Schur complement (and get M factor for recovering v later)
            S, M_fact = self.compute_S()

            # reduced normal eqns: (A^T S A) delta = - A^T S w
            N = self.A.T @ S @ self.A                 # 3x3
            rhs = - self.A.T @ S @ self.w             # 3x1

            # solve for delta_theta (Cholesky if positive-def)
            try:
                cfN = cho_factor(N, overwrite_a=False, check_finite=False)
                delta_theta = cho_solve(cfN, rhs, check_finite=False)
            except Exception:
                print("Matrix not pos-def, using np.linalg.solve")
                delta_theta = np.linalg.solve(N + 1e-12*np.eye(3), rhs)

            # update theta
            self.theta = self.theta + delta_theta

            if np.linalg.norm(delta_theta) < tol:
                if verbose:
                    print("Converged.")
                break

            residual_term = (self.A @ delta_theta) + self.w   # 3n x 1
            v = self.recover_v(residual_term, M_fact)

            self.delta_theta = delta_theta
            self.v = v
            self.S = S
            self.M_fact = M_fact


            if verbose:
                print(f"[iter {it+1}] Δθ = {delta_theta.flatten()*180/np.pi} [deg]")
                print(f"[iter {it+1}] θ = {self.theta.flatten()*180/np.pi} [deg]")
                cond_residual = self.A @ delta_theta + self.B @ self.v + self.w
                rms_cond = np.sqrt(np.mean(np.linalg.norm(cond_residual, axis=1)**2))
                print(f"[iter {it+1}] Mean residual: {rms_cond:.4f} m")


        return self.theta, self.v
    
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
