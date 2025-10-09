import numpy as np
from lib.rotations import *
from matplotlib import pyplot as plt


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
    def __init__(self, rawCor, trj, mount, R_e2enu, sigmas):
        poses_i = trj.interpolate(rawCor[:, 0], customRPY=True)
        poses_j = trj.interpolate(rawCor[:, 1], customRPY=True)

        self.corSet = []
        for k in range(len(rawCor)):
            self.corSet.append(Correspondence(rawCor[k], poses_i[k], poses_j[k], mount, R_e2enu))
            self.corSet[k].compute_l_hat()
            self.corSet[k].compute_Rb2m()
            self.corSet[k].computeA()
            self.corSet[k].computeB(mount['bor'])
            self.corSet[k].compute_w(mount['bor'])

        self.sigmas = sigmas

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
         
        assert Pk.shape == (12,12)
        n = len(self.corSet)
        self.P = np.zeros((12*n, 12*n))
        for k in range(n):
            self.P[12*k:12*(k+1), 12*k:12*(k+1)] = Pk
    
    def plotP(self):
        plt.imshow(self.P, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Covariance Matrix P')
        plt.xlabel(f"Size: {self.P.shape[0]} x {self.P.shape[1]}")
        plt.show()
        #show only one Pk
        Pk = self.P[0:12, 0:12]
        plt.imshow(Pk, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title('Covariance Matrix Pk')
        plt.xlabel(f"Size: {Pk.shape[0]} x {Pk.shape[1]}")
        plt.show()

    def plotResiduals(self):
        res = np.hstack([c.w for c in self.corSet])
        print("Current residuals stats:")
        print(f"Mean residual: {np.mean(np.linalg.norm(res, axis=0))} m")
        print(f"Median residual: {np.median(np.linalg.norm(res, axis=0))} m")
        print(f"Max residual: {np.max(np.linalg.norm(res, axis=0))} m") 
        plt.hist(np.linalg.norm(res, axis=0), bins=50)
        plt.xlabel('Residual norm (m)')
        plt.ylabel('Count')
        plt.title('Histogram of correspondence residuals')
        plt.grid()
        plt.show()