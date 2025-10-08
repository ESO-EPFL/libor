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

    def compute_w(self, theta):
        borVector_i = self.U_i @ theta
        borVector_j = self.U_j @ theta

        self.p_i = self.l_hat[0:3].reshape(3,1) + self.R_b2m_i @ (self.u_i + borVector_i + self.leverArm)
        self.p_j = self.l_hat[6:9].reshape(3,1) + self.R_b2m_j @ (self.u_j + borVector_j + self.leverArm)

        print("p_i:", self.p_i)
        print("p_j:", self.p_j)

        self.w = self.p_i - self.p_j

    def computeP(self, sigma_xy, sigma_z, sigma_rp, sigma_y):
        #TODO Put in constructor
        self.P = np.diag([sigma_xy**2, sigma_xy**2, sigma_z**2, sigma_rp**2, sigma_rp**2, sigma_y**2,
                          sigma_xy**2, sigma_xy**2, sigma_z**2, sigma_rp**2, sigma_rp**2, sigma_y**2])
        
    # def computeResStats(self):
    #     dist = np.linalg.norm(self.w)

    #     mean = np.mean(dist)
    #     median = np.median(dist)
    #     q25 = np.percentile(dist, 25)
    #     q75 = np.percentile(dist, 75)
    #     std = np.std(dist)
    #     max = np.max(dist)
        
    #     plt.hist(dist, bins=50)
    #     plt.title(f"Residual: mean={mean:.3f}, q25={q25:.3f}, q50={median:.3f}, q75={q75:.3f}, std={std:.3f}, max={max:.3f} (m)")
    #     plt.xlabel("Distance (m)")
    #     plt.ylabel("Count")
    #     plt.show()