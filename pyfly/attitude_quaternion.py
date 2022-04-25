import numpy as np


class AttitudeQuaternion:
    def __init__(self):
        """
        Quaternion attitude representation used by PyFly.
        """
        self.quaternion = None
        self.euler_angles = {"roll": None, "pitch": None, "yaw": None}
        self.history = None

    def seed(self, seed):
        return

    def reset(self, euler_init):
        """
        Reset state of attitude quaternion to value given by euler angles.

        :param euler_init: ([float]) the roll, pitch, yaw values to initialize quaternion to.
        """
        if euler_init is not None:
            self._from_euler_angles(euler_init)
        else:
            raise NotImplementedError
        self.history = [self.quaternion]

    def as_euler_angle(self, angle="all", timestep=-1):
        """
        Get attitude quaternion as euler angles, roll, pitch and yaw.

        :param angle: (string) which euler angle to return or all.
        :param timestep: (int) timestep
        :return: (float or dict) requested euler angles.
        """
        e0, e1, e2, e3 = self.history[timestep]
        res = {}
        if angle == "roll" or angle == "all":
            res["roll"] = np.arctan2(2 * (e0 * e1 + e2 * e3), e0 ** 2 + e3 ** 2 - e1 ** 2 - e2 ** 2)
        if angle == "pitch" or angle == "all":
            res["pitch"] = np.arcsin(2 * (e0 * e2 - e1 * e3))
        if angle == "yaw" or angle == "all":
            res["yaw"] = np.arctan2(2 * (e0 * e3 + e1 * e2), e0 ** 2 + e1 ** 2 - e2 ** 2 - e3 ** 2)

        return res if angle == "all" else res[angle]

    @property
    def value(self):
        return self.quaternion

    def _from_euler_angles(self, euler):
        """
        Set value of attitude quaternion from euler angles.

        :param euler: ([float]) euler angles roll, pitch, yaw.
        """
        phi, theta, psi = euler
        e0 = np.cos(psi / 2) * np.cos(theta / 2) * np.cos(phi / 2) + np.sin(psi / 2) * np.sin(theta / 2) * np.sin(
            phi / 2)
        e1 = np.cos(psi / 2) * np.cos(theta / 2) * np.sin(phi / 2) - np.sin(psi / 2) * np.sin(theta / 2) * np.cos(
            phi / 2)
        e2 = np.cos(psi / 2) * np.sin(theta / 2) * np.cos(phi / 2) + np.sin(psi / 2) * np.cos(theta / 2) * np.sin(
            phi / 2)
        e3 = np.sin(psi / 2) * np.cos(theta / 2) * np.cos(phi / 2) - np.cos(psi / 2) * np.sin(theta / 2) * np.sin(
            phi / 2)

        self.quaternion = (e0, e1, e2, e3)

    def set_value(self, quaternion, save=True):
        """
        Set value of attitude quaternion.

        :param quaternion: ([float]) new quaternion value
        :param save: (bool) whether to commit value to history of attitude.
        """
        self.quaternion = quaternion
        if save:
            self.history.append(self.quaternion)