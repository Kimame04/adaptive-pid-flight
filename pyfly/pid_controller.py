import numpy as np
import pandas as pd
import openpyxl
import matplotlib.pyplot as plt

from algo.RLS import RLS
from algo.recursiveLeastSquares import RecursiveLeastSquares


class PIDController:
    def __init__(self, dt=0.01):

        self.k_p_V = 0.5
        self.k_i_V = 0.1
        self.k_p_phi = 1
        self.k_i_phi = 0  # Note gain is set to zero for roll channel
        self.k_d_phi = 0.5
        self.k_p_theta = -4
        self.k_i_theta = -0.75
        self.k_d_theta = -0.1

        self.delta_a_min = np.radians(-30)
        self.delta_e_min = np.radians(-30)
        self.delta_a_max = np.radians(30)
        self.delta_e_max = np.radians(35)

        self.dt = dt

        self.va_r = None
        self.phi_r = None
        self.theta_r = None

        self.de = None

        self.int_va = 0
        self.int_roll = 0
        self.int_pitch = 0

        self.newA = None
        self.newB = None
        self.newC = None
        self.newD = None
        self.count = 0

        self.ALPHA = 3.5
        self.BETA = 4.8
        self.sum_pitch_error = 0
        self.sum_roll_error = 0

        self.df = pd.DataFrame(columns=list('PID'))

    def set_reference(self, phi, theta, va):
        self.va_r = va
        self.phi_r = phi
        self.theta_r = theta

    def reset(self):
        self.int_va = 0
        self.int_roll = 0
        self.int_pitch = 0

    def get_action(self, phi, theta, va, omega, isAdaptive):
        
        e_V_a = va - self.va_r
        e_phi = phi - self.phi_r
        e_theta = theta - self.theta_r

        self.sum_pitch_error += abs(e_theta)
        self.sum_roll_error += abs(e_phi)

        #adaptive

        self.count += 1

        if(isAdaptive):


            if (self.count <= 3 and self.count > 1):

                if (self.count == 2):
                    self.newA = np.array([[self.k_p_theta, self.k_i_theta]])
                    self.newB = np.array([[e_theta]])

                    self.newC = np.array([[self.k_p_phi, self.k_i_phi]])
                    self.newD = np.array([[e_phi]])
                else:
                    self.newA = np.vstack((self.newA, np.array([self.k_p_theta, self.k_i_theta])))
                    self.newB = np.vstack((self.newB, np.array(e_theta)))
                    self.rls_pitch = RecursiveLeastSquares(self.newA, self.newB)

                    self.newC = np.vstack((self.newC, np.array([self.k_p_phi, self.k_i_phi])))
                    self.newD = np.vstack((self.newD, np.array(e_phi)))
                    self.rls_roll = RecursiveLeastSquares(self.newC, self.newD)

                self.k_p_theta += 0.001
                self.k_i_theta += 0.001

                self.k_p_phi += 0.001
                self.k_i_phi += 0.001

            elif self.count > 3:

                self.rls_pitch.addData(np.array([self.k_p_theta, self.k_i_theta]), np.array([e_theta]))
                self.k_p_theta = -abs(self.rls_pitch.x[0][0]) * self.ALPHA * self.count
                self.k_i_theta = -abs(self.rls_pitch.x[1][0]) * self.ALPHA * self.count

                self.rls_roll.addData(np.array([self.k_p_phi, self.k_i_phi]), np.array([e_phi]))
                #self.k_p_phi = -abs(self.rls_roll.x[0][0]) * self.BETA * np.clip(self.count, 0, 10)
                #self.k_i_phi = -abs(self.rls_roll.x[1][0]) * self.BETA * np.clip(self.count, 0, 10)

        # (integral states are initialized to zero)
        self.int_va = self.int_va + self.dt * e_V_a
        self.int_roll = self.int_roll + self.dt * e_phi
        self.int_pitch = self.int_pitch + self.dt * e_theta

        # Note the different sign on pitch gains below.
        # Positive aileron  -> positive roll moment
        # Positive elevator -> NEGATIVE pitch moment

        delta_t = 0 - self.k_p_V * e_V_a - self.k_i_V * self.int_va  # PI
        delta_a = - self.k_p_phi * e_phi - self.k_i_phi * self.int_roll - self.k_d_phi * omega[0]  # PID
        delta_e = 0 - self.k_p_theta * e_theta - self.k_i_theta * self.int_pitch - self.k_d_theta * omega[1]  # PID
        delta_r = 0  # No rudder available

        # Constrain input
        delta_t = np.clip(delta_t, 0, 1.0)  # throttle between 0 and 1

        delta_a = np.clip(delta_a, self.delta_a_min, self.delta_a_max)
        delta_e = np.clip(delta_e, self.delta_e_min, self.delta_e_max)

        return np.asarray([delta_e, delta_a, delta_t])


        

