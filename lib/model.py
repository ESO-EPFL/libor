import numpy as np
from lib.rotations import *


class Correspondence:
    def __init__(self, corr, trj, mount, R_e2enu):
        self.t_i = corr[0]
        self.t_j = corr[1]

        self.R_s2b = mount.R_s2b
        self.leverArm = mount.leverArm

        xyz_i, _, R_ned2e_i, rpy_i = trj.interpolate(self.t_i, customRPY=True)
        xyz_j, _, R_ned2e_j, rpy_j = trj.interpolate(self.t_j, customRPY=True)

        self.l_ij = np.hstack((xyz_i, rpy_i, xyz_j, rpy_j)).reshape(12,1)
        self.l_ij_hat = np.empty((12,1))

        self.u_i = self.R_s2b @ corr[2:5].reshape(3,1)
        self.u_j = self.R_s2b @ corr[5:8].reshape(3,1)

        self.U_i = skewT(self.u_i)
        self.U_j = skewT(self.u_j)

        self.v_ij = np.zeros((12,1))

        self.R_e2enu = R_e2enu

        self.R_ned2e_i = R_ned2e_i
        self.R_ned2e_j = R_ned2e_j 

        self.A = np.zeros((3,3))
        self.B = np.zeros((3,12))
        self.w = np.zeros((3,1))
        self.p_i = np.zeros((3,1))
        self.p_j = np.zeros((3,1))
        self.P = np.zeros((12,12))

    def l_ij_hat(self):
        self.l_ij_hat =  self.l_ij + self.v_ij

    def compute_Rb2m(self):
        r_i, p_i, y_i = self.l_ij_hat[3:6]
        r_j, p_j, y_j = self.l_ij_hat[9:12]

        R_b2ned_i = R_b2ned(r_i, p_i, y_i)
        R_b2ned_j = R_b2ned(r_j, p_j, y_j)

        self.R_b2m_i = self.R_e2enu @ self.R_ned2e_i @ R_b2ned_i
        self.R_b2m_j = self.R_e2enu @ self.R_ned2e_j @ R_b2ned_j

    def computeA(self):
        self.A = self.R_b2m_i @ self.U_i - self.R_b2m_j @ self.U_j

    def computeB(self, theta):

        r_i, p_i, y_i = self.l_ij_hat[3:6]
        r_j, p_j, y_j = self.l_ij_hat[9:12]

        s_i = self.u_i + self.U_i @ theta + self.mount.leverArm
        s_j = self.u_j + self.U_j @ theta + self.mount.leverArm

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

        self.p_i = np.squeeze(self.l_ij_hat[0:3].reshape(3,1) + self.R_b2m_i @ (self.u_i + borVector_i + self.leverArm))
        self.p_j = np.squeeze(self.l_ij_hat[6:9].reshape(3,1) + self.R_b2m_j @ (self.u_j + borVector_j + self.leverArm))

        self.w = self.p_i - self.p_j

    def computeP(self, sigma_xy, sigma_z, sigma_rp, sigma_y):
        self.P = np.diag([sigma_xy**2, sigma_xy**2, sigma_z**2, sigma_rp**2, sigma_rp**2, sigma_y**2,
                          sigma_xy**2, sigma_xy**2, sigma_z**2, sigma_rp**2, sigma_rp**2, sigma_y**2])