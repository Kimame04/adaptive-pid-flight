from matplotlib import pyplot as plt
import numpy as np

from constraint_exception import ConstraintException


class Variable:
    def __init__(self, name, value_min=None, value_max=None, init_min=None, init_max=None, constraint_min=None,
                 constraint_max=None, convert_to_radians=False, unit=None, label=None, wrap=False):
        """
        PyFly state object managing state history, constraints and visualizations.

        :param name: (string) name of state
        :param value_min: (float) lowest possible value of state, values will be clipped to this limit
        :param value_max: (float) highest possible value of state, values will be clipped to this limit
        :param init_min: (float) lowest possible initial value of state
        :param init_max: (float) highest possible initial value of state
        :param constraint_min: (float) lower constraint of state, which if violated will raise ConstraintException
        :param constraint_max: (float) upper constraint of state, which if violated will raise ConstraintException
        :param convert_to_radians: (bool) whether to convert values for attributes from configuration file from degrees
        to radians
        :param unit: (string) unit of the state, for plotting purposes
        :param label: (string) label given to state in plots
        :param wrap: (bool) whether to wrap state value in region [-pi, pi]
        """
        self.value_min = value_min
        self.value_max = value_max

        self.init_min = init_min if init_min is not None else value_min
        self.init_max = init_max if init_max is not None else value_max

        self.constraint_min = constraint_min
        self.constraint_max = constraint_max

        if convert_to_radians:
            for attr_name, val in self.__dict__.items():
                if val is not None:
                    setattr(self, attr_name, np.radians(val))

        self.name = name

        self.value = None

        self.wrap = wrap

        self.unit = unit
        self.label = label if label is not None else self.name
        self.lines = {"self": None}
        self.target_lines = {"self": None}
        self.target_bounds = {"self": None}

        self.np_random = None
        self.seed()

        self.history = None

    def reset(self, value=None):
        """
        Reset object to initial state.

        :param value: (float) initial value of state
        """
        self.history = []

        if value is None:
            try:
                value = self.np_random.uniform(self.init_min, self.init_max)
            except TypeError:
                raise Exception("Variable init_min and init_max can not be None if no value is provided on reset")
        else:
            value = self.apply_conditions(value)

        self.value = value

        self.history.append(value)

    def seed(self, seed=None):
        """
        Seed random number generator of state

        :param seed: (int) seed of random state
        """
        self.np_random = np.random.RandomState(seed)

    def apply_conditions(self, value):
        """
        Apply state limits and constraints to value. Will raise ConstraintException if constraints are violated

        :param value: (float) value to which limits and constraints are applied
        :return: (float) value after applying limits and constraints
        """
        if self.constraint_min is not None and value < self.constraint_min:
            raise ConstraintException(self.name, value, self.constraint_min)

        if self.constraint_max is not None and value > self.constraint_max:
            raise ConstraintException(self.name, value, self.constraint_max)

        if self.value_min is not None or self.value_max is not None:
            value = np.clip(value, self.value_min, self.value_max)

        if self.wrap and np.abs(value) > np.pi:
            value = np.sign(value) * (np.abs(value) % np.pi - np.pi)

        return value

    def set_value(self, value, save=True):
        """
        Set value of state, after applying limits and constraints to value. Raises ConstraintException if constraints
        are violated

        :param value: (float) new value of state
        :param save: (bool) whether to commit value to history of state
        """
        value = self.apply_conditions(value)

        if save:
            self.history.append(value)

        self.value = value

    def plot(self, axis=None, y_unit=None, target=None, plot_id=None, **plot_kw):
        """
        Plot state history.

        :param axis: (matplotlib.pyplot.axis or None) axis object to plot state to. If None create new axis
        :param y_unit: (string) unit state should be plotted in, will convert values if different from internal
        representation
        :param target: (list) target values for state, must be of equal size to state history
        :param plot_id: (string or int or None) identifier of parent plot object. Allows state to plot to multiple
        figures at a time.
        :param plot_kw: (dict) plot keyword arguments passed to matplotlib.pyplot.plot
        """

        def linear_scaling(val, old_min, old_max, new_min, new_max):
            return (new_max - np.sign(old_min) * (- new_min)) / (old_max - old_min) * (
                        np.array(val) - old_max) + new_max

        if y_unit is None:
            y_unit = self.unit if y_unit is None else y_unit

        x, y = self._get_plot_x_y_data()
        if "degrees" in y_unit:
            y = np.degrees(y)
            if target is not None:
                target["data"] = np.degrees(target["data"])
                if "bound" in target:
                    target["bound"] = np.degrees(target["bound"])
        elif y_unit == "%":  # TODO: scale positive according to positive limit and negative according to lowest minimum value
            y = linear_scaling(y, self.value_min, self.value_max, -100, 100)
            if target is not None:
                target["data"] = linear_scaling(target["data"], self.value_min, self.value_max, -100, 100)
                if "bound" in target:
                    target["bound"] = linear_scaling(target["bound"], self.value_min, self.value_max, -100, 100)
        else:
            y = y

        plot_object = axis
        if axis is None:
            plot_object = plt
            plot_id = "self"
            fig_kw = {"title": self.name, "ylabel": y_unit}

        if self.lines.get(plot_id, None) is None:
            line, = plot_object.plot(x, y, label=self.label, **plot_kw)
            self.lines[plot_id] = line

            if target is not None:
                tar_line, = plot_object.plot(x, target["data"], color=self.lines[plot_id].get_color(), linestyle="dashed",
                                             marker="x", markevery=0.2)

                if "bound" in target:
                    tar_bound = plot_object.fill_between(np.arange(target["bound"].shape[0]),
                                                         target["data"] + target["bound"],
                                                         target["data"] - target["bound"], alpha=0.15,
                                                         facecolor=self.lines[plot_id].get_color()
                                                        )
                    self.target_bounds[plot_id] = tar_bound
                self.target_lines[plot_id] = tar_line
        else:
            self.lines[plot_id].set_data(x, y)
            if target is not None:
                self.target_lines[plot_id].set_data(x, target)
                if "bound" in target:  # TODO: fix this?
                    self.target_bounds[plot_id].set_data(np.arange(target["bound"].shape[0]),
                                                         target["data"] + target["bound"],
                                                         target["data"] - target["bound"])
        if axis is None:
            for k, v in fig_kw.items():
                getattr(plot_object, format(k))(v)
            plt.show()

    def close_plot(self, plot_id="self"):
        """
        Close plot with id plot_id.

        :param plot_id: (string or int) identifier of parent plot object
        """
        self.lines[plot_id] = None
        self.target_lines[plot_id] = None
        self.target_bounds[plot_id] = None

    def _get_plot_x_y_data(self):
        """
        Get plot data from variable history.

        :return: ([int], [float]) x plot data, y plot data
        """
        x = list(range(len(self.history)))
        y = self.history
        return x, y
