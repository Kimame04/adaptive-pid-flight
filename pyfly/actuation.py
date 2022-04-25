import numpy as np


class Actuation:
    def __init__(self, model_inputs, actuator_inputs, dynamics):
        """
        PyFly actuation object, responsible for verifying validity of configured actuator model, processing inputs and
        actuator dynamics.

        :param model_inputs: ([string]) the states used by PyFly as inputs to dynamics
        :param actuator_inputs: ([string]) the user configured actuator input states
        :param dynamics: ([string]) the user configured actuator states to simulate dynamics for
        """
        self.states = {}
        self.coefficients = [[np.array([]) for _ in range(3)] for __ in range(2)]
        self.elevon_dynamics = False
        self.dynamics = dynamics
        self.inputs = actuator_inputs
        self.model_inputs = model_inputs
        self.input_indices = {s: i for i, s in enumerate(actuator_inputs)}
        self.dynamics_indices = {s: i for i, s in enumerate(dynamics)}

    def set_states(self, values, save=True):
        """
        Set values of actuator states.

        :param values: ([float]) list of state values + list of state derivatives
        :param save: (bool) whether to commit values to state history
        :return:
        """
        for i, state in enumerate(self.dynamics):
            self.states[state].set_value((values[i], values[len(self.dynamics) + i]), save=save)

        # Simulator model operates on elevator and aileron angles, if aircraft has elevon dynamics need to map
        if self.elevon_dynamics:
            elevator, aileron = self._map_elevon_to_elevail(er=self.states["elevon_right"].value,
                                                            el=self.states["elevon_left"].value)

            self.states["aileron"].set_value((aileron, 0), save=save)
            self.states["elevator"].set_value((elevator, 0), save=save)

    def add_state(self, state):
        """
        Add actuator state, and configure dynamics if state has dynamics.

        :param state: (ControlVariable) actuator state
        :return:
        """
        self.states[state.name] = state
        if state.name in self.dynamics:
            for i in range(2):
                for j in range(3):
                    self.coefficients[i][j] = np.append(self.coefficients[i][j], state.coefs[i][j])

    def get_values(self):
        """
        Get state values and derivatives for states in actuator dynamics.

        :return: ([float]) list of state values + list of state derivatives
        """
        return [self.states[state].value for state in self.dynamics] + [self.states[state].dot for state in
                                                                            self.dynamics]

    def rhs(self, setpoints=None):
        """
        Right hand side of actuator differential equation.

        :param setpoints: ([float] or None) setpoints for actuators. If None, setpoints are set as the current command
        of the dynamics variable
        :return: ([float]) right hand side of actuator differential equation.
        """
        if setpoints is None:
            setpoints = [self.states[state].command for state in self.dynamics]
        states = [self.states[state].value for state in self.dynamics]
        dots = [self.states[state].dot for state in self.dynamics]
        dot = np.multiply(states,
                          self.coefficients[0][0]) + np.multiply(setpoints,
                                                                 self.coefficients[0][2]) + np.multiply(dots, self.coefficients[0][1])
        ddot = np.multiply(states,
                           self.coefficients[1][0]) + np.multiply(setpoints,
                                                                  self.coefficients[1][2]) + np.multiply(dots, self.coefficients[1][1])

        return np.concatenate((dot, ddot))

    def set_and_constrain_commands(self, commands):
        """
        Take  raw actuator commands and constrain them according to the state limits and constraints, and update state
        values and history.

        :param commands: ([float]) raw commands
        :return: ([float]) constrained commands
        """
        dynamics_commands = {}
        if self.elevon_dynamics and "elevator" and "aileron" in self.inputs:
            elev_c, ail_c = commands[self.input_indices["elevator"]], commands[self.input_indices["aileron"]]
            elevon_r_c, elevon_l_c = self._map_elevail_to_elevon(elev=elev_c, ail=ail_c)
            dynamics_commands = {"elevon_right": elevon_r_c, "elevon_left": elevon_l_c}

        for state in self.dynamics:
            if state in self.input_indices:
                state_c = commands[self.input_indices[state]]
            else:  # Elevail inputs with elevon dynamics
                state_c = dynamics_commands[state]
            self.states[state].set_command(state_c)
            dynamics_commands[state] = self.states[state].command

        # The elevator and aileron commands constrained by limitatons on physical elevons
        if self.elevon_dynamics:
            elev_c, ail_c = self._map_elevon_to_elevail(er=dynamics_commands["elevon_right"],
                                                        el=dynamics_commands["elevon_left"])
            self.states["elevator"].set_command(elev_c)
            self.states["aileron"].set_command(ail_c)

        for state, i in self.input_indices.items():
            commands[i] = self.states[state].command

        return commands

    def finalize(self):
        """
        Assert valid configuration of actuator dynamics and set actuator state limits if applicable.
        """
        if "elevon_left" in self.dynamics or "elevon_right" in self.dynamics:
            assert("elevon_left" in self.dynamics and "elevon_right" in self.dynamics and not ("aileron" in self.dynamics
                   or "elevator" in self.dynamics))
            assert ("elevon_left" in self.states and "elevon_right" in self.states)
            self.elevon_dynamics = True

            # Set elevator and aileron limits from elevon limits for plotting purposes etc.
            if "elevator" in self.states:
                elev_min, _ = self._map_elevon_to_elevail(er=self.states["elevon_right"].value_min,
                                                          el=self.states["elevon_left"].value_min)
                elev_max, _ = self._map_elevon_to_elevail(er=self.states["elevon_right"].value_max,
                                                          el=self.states["elevon_left"].value_max)
                self.states["elevator"].value_min = elev_min
                self.states["elevator"].value_max = elev_max
            if "aileron" in self.states:
                _, ail_min = self._map_elevon_to_elevail(er=self.states["elevon_right"].value_max,
                                                         el=self.states["elevon_left"].value_min)
                _, ail_max = self._map_elevon_to_elevail(er=self.states["elevon_right"].value_min,
                                                         el=self.states["elevon_left"].value_max)
                self.states["aileron"].value_min = ail_min
                self.states["aileron"].value_max = ail_max

    def reset(self, state_init=None):
        for state in self.dynamics:
            init = None
            if state_init is not None and state in state_init:
                init = state_init[state]
            self.states[state].reset(value=init)

        if self.elevon_dynamics:
            elev, ail = self._map_elevon_to_elevail(er=self.states["elevon_right"].value, el=self.states["elevon_left"].value)
            self.states["elevator"].reset(value=elev)
            self.states["aileron"].reset(value=ail)

    def _map_elevail_to_elevon(self, elev, ail):
        er = -1 * ail + elev
        el = ail + elev
        return er, el

    def _map_elevon_to_elevail(self, er, el):
        ail = (-er + el) / 2
        elev = (er + el) / 2
        return elev, ail
