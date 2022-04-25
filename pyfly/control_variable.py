
import numpy as np
from variable import Variable


class ControlVariable(Variable):
    def __init__(self, order=None, tau=None, omega_0=None, zeta=None, dot_max=None, disabled=False, **kwargs):
        """
        PyFly actuator state variable.

        :param order: (int) order of state transfer function
        :param tau: (float) time constant for first order transfer functions
        :param omega_0: (float) undamped natural frequency of second order transfer functions
        :param zeta: (float) damping factor of second order transfer function
        :param dot_max: (float) constraint on magnitude of derivative of second order transfer function
        :param disabled: (bool) if actuator is disabled for aircraft, e.g. aircraft has no rudder
        :param kwargs: (dict) keyword arguments for Variable class
        """
        assert (disabled or (order == 1 or order == 2))
        super().__init__(**kwargs)
        self.order = order
        self.tau = tau
        self.omega_0 = omega_0
        self.zeta = zeta
        self.dot_max = dot_max

        if order == 1:
            assert (tau is not None)
            self.coefs = [[-1 / self.tau, 0, 1 / self.tau], [0, 0, 0]]
        elif order == 2:
            assert (omega_0 is not None and zeta is not None)
            self.coefs = [[0, 1, 0], [-self.omega_0 ** 2, -2 * self.zeta * self.omega_0, self.omega_0 ** 2]]
        self.dot = None
        self.command = None
        self.disabled = disabled
        if self.disabled:
            self.value = 0
        self.plot_quantity = "value"

    def apply_conditions(self, values):
        """
        Apply state limits and constraints to value. Will raise ConstraintException if constraints are violated

        :param value: (float) value to which limits and constraints is applied
        :return: (float) value after applying limits and constraints
        """
        try:
            value, dot = values
        except:
            value, dot = values, 0
        value = super().apply_conditions(value)

        if self.dot_max is not None:
            dot = np.clip(dot, -self.dot_max, self.dot_max)

        return [value, dot]

    def set_command(self, command):
        """
        Set setpoint for actuator and commit to history of state

        :param command: setpoint for actuator
        """
        command = super().apply_conditions(command)
        self.command = command
        self.history["command"].append(command)

    def reset(self, value=None):
        """
        Reset object to initial state.

        :param value: (list) initial value, derivative and setpoint of state
        """
        self.history = {"value": [], "dot": [], "command": []}

        if not self.disabled:
            if value is None:
                value = self.np_random.uniform(self.init_min, self.init_max), 0
            else:
                value = self.apply_conditions(value)

            self.value = value[0]
            self.dot = value[1]
            command = None
            self.command = command
        else:
            value, dot, command = 0, 0, None
            self.value = value
            self.dot = dot
            self.command = command

        self.history["value"].append(self.value)
        self.history["dot"].append(self.dot)

    def set_value(self, value, save=True):
        """
        Set value of state, after applying limits and constraints to value. Raises ConstraintException if constraints
        are violated

        :param value: (float) new value and derivative of state
        :param save: (bool) whether to commit value to history of state
        """
        value, dot = self.apply_conditions(value)

        self.value = value
        self.dot = dot

        if save:
            self.history["value"].append(value)
            self.history["dot"].append(dot)

    def _get_plot_x_y_data(self):
        """
        Get plot data from variable history, for the quantity designated by the attribute plot_quantity.

        :return: ([int], [float]) x plot data, y plot data
        """
        y = self.history[self.plot_quantity]
        x = list(range(len(y)))
        return x, y

    def get_coeffs(self):
        if self.order == 1:
            return
        else:
            return []