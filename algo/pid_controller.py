import numpy as np
import pandas as pd
import openpyxl

from recursiveLeastSquares import RecursiveLeastSquares


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
        self.temp = None

        self.int_va = 0
        self.int_roll = 0
        self.int_pitch = 0

        self.newA = None
        self.newB = None
        self.count = 0

        self.df = pd.DataFrame(columns=list('PIDE'))
        self.df2 = pd.DataFrame(columns=list('EAT'))

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

        #adaptive

        if(isAdaptive):

            self.count += 1

            if (self.count == 1):
                self.temp = e_theta

            elif (self.count <= 3):

                self.de = 1 - abs(self.temp - e_theta)
                self.temp = e_theta

                if (self.count == 2):
                    #self.newA = np.array([[self.k_p_theta,self.k_i_theta, self.k_d_theta]])
                    self.newA = np.array([[self.k_p_theta, self.k_i_theta]])
                    self.newB = np.array([[e_theta]])
                else:
                    #self.newA = np.vstack((self.newA, np.array([self.k_p_theta,self.k_i_theta, self.k_d_theta])))
                    self.newA = np.vstack((self.newA, np.array([self.k_p_theta, self.k_i_theta])))
                    self.newB = np.vstack((self.newB, np.array([self.de])))
                    self.rls = RecursiveLeastSquares(self.newA, self.newB)

                self.df = self.df.append({'P': self.k_p_theta, 'I': self.k_i_theta, 'D': self.k_d_theta, 'E': e_theta}, ignore_index=True)

                self.k_p_theta += 0.001
                self.k_i_theta += 0.001
                self.k_d_theta += 0.001
            else:
                self.de = 1 - abs(self.temp - e_theta)
                self.temp = e_theta
                #print(self.newA)
                #print(self.newB)
                #self.rls.addData(np.array([self.k_p_theta,self.k_i_theta, self.k_d_theta]), np.array([e_theta]))
                self.rls.addData(np.array([self.k_p_theta, self.k_i_theta]), np.array([e_theta]))
                #print(rls.x)
                self.k_p_theta = self.rls.x[0][0]
                self.k_i_theta = self.rls.x[1][0]
                #self.k_d_theta = self.rls.x[2][0]
                self.df = self.df.append({'P': self.k_p_theta, 'I': self.k_i_theta, 'D': self.k_d_theta, 'E': e_theta}, ignore_index=True)
                #print(rls.A)

        self.df2 = self.df2.append({'E': delta_e, 'A': delta_a, 'T': delta_t, 'E': e_theta},
                                 ignore_index=True)

        return np.asarray([delta_e, delta_a, delta_t])


        

