import numpy as np
from algo.dryden import DrydenGustModel


class Wind:
    def __init__(self, turbulence, mag_min=None, mag_max=None, b=None, turbulence_intensity=None, sim_length=300, dt=None):
        """
        Wind and turbulence object used by PyFly.

        :param turbulence: (bool) whether turbulence is enabled
        :param mag_min: (float) minimum magnitude of steady wind component
        :param mag_max: (float) maximum magnitude of steady wind component
        :param b: (float) wingspan of aircraft
        :param turbulence_intensity: (string) intensity of turbulence
        :param dt: (float) integration step length
        """
        self.turbulence = turbulence
        self.mag_min = mag_min
        self.mag_max = mag_max
        self.steady = None
        self.components = []
        self.turbulence_sim_length = sim_length

        if self.turbulence:
            self.dryden = DrydenGustModel(self.turbulence_sim_length, dt, b, intensity=turbulence_intensity)
        else:
            self.dryden = None

        self.np_random = None
        self.seed()

    def seed(self, seed=None):
        """
        Seed random number generator of object

        """
        self.np_random = np.random.RandomState(seed)
        if self.turbulence:
            self.dryden.seed(seed)

    def reset(self, value=None, noise=None):
        """
        Reset wind object to initial state

        :param value: ([float] or float) strength and direction of the n, e and d components or magnitude of the steady wind.
        """
        if value is None or isinstance(value, float) or isinstance(value, int):
            if value is None and self.mag_min is None and self.mag_max is None:
                value = []
                for comp in self.components:
                    comp.reset()
                    value.append(comp.value)
            else:
                if value is None:
                    magnitude = self.np_random.uniform(self.mag_min, self.mag_max)
                else:
                    magnitude = value
                w_n = self.np_random.uniform(-magnitude, magnitude)
                w_e_max = np.sqrt(magnitude ** 2 - w_n ** 2)
                w_e = self.np_random.uniform(-w_e_max, w_e_max)
                w_d = np.sqrt(magnitude ** 2 - w_n ** 2 - w_e ** 2)
                value = [w_n, w_e, w_d]

        if self.turbulence:
            self.dryden.reset(noise)

        self.steady = value
        for i, comp in enumerate(self.components):
            comp.reset(value[i])

    def set_value(self, timestep):
        """
        Set value to wind value at timestep t

        :param timestep: (int) timestep
        """
        value = self.steady

        if self.turbulence:
            value += self._get_turbulence(timestep, "linear")

        for i, comp in enumerate(self.components):
            comp.set_value(value[i])

    def get_turbulence_linear(self, timestep):
        """
        Get linear component of turbulence model at given timestep

        :param timestep: (int) timestep
        :return: ([float]) linear component of turbulence at given timestep
        """
        return self._get_turbulence(timestep, "linear")

    def get_turbulence_angular(self, timestep):
        """
        Get angular component of turbulence model at given timestep

        :param timestep: (int) timestep
        :return: ([float]) angular component of turbulence at given timestep
        """
        return self._get_turbulence(timestep, "angular")

    def _get_turbulence(self, timestep, component):
        """
        Get turbulence at given timestep.

        :param timestep: (int) timestep
        :param component: (string) which component to return, linear or angular.
        :return: ([float]) turbulence component at timestep
        """
        if timestep >= self.dryden.sim_length:
            self.dryden.simulate(self.turbulence_sim_length)

        if component == "linear":
            return self.dryden.vel_lin[:, timestep]
        else:
            return self.dryden.vel_ang[:, timestep]

