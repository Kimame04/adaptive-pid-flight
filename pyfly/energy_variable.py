from variable import Variable


class EnergyVariable(Variable):
    def __init__(self, mass=None, inertia_matrix=None, gravity=None, **kwargs):
        super().__init__(**kwargs)
        self.required_variables = []
        self.variables = {}
        if self.name == "energy_potential" or self.name == "energy_total":
            assert(mass is not None and gravity is not None)
            self.mass = mass
            self.gravity = gravity
            self.required_variables.append("position_d")
        if self.name == "energy_kinetic" or self.name == "energy_total":
            assert (mass is not None and inertia_matrix is not None)
            self.mass = mass
            self.inertia_matrix = inertia_matrix
            self.required_variables.extend(["Va", "omega_p", "omega_q", "omega_r"])
        if self.name == "energy_kinetic_rotational":
            assert(inertia_matrix is not None)
            self.inertia_matrix = inertia_matrix
            self.required_variables.extend(["omega_p", "omega_q", "omega_r"])
        if self.name == "energy_kinetic_translational":
            assert(mass is not None)
            self.mass = mass
            self.required_variables.append("Va")

    def add_requirement(self, name, variable):
        self.variables[name] = variable

    def calculate_value(self):
        val = 0
        if self.name == "energy_potential" or self.name == "energy_total":
            val += self.mass * self.gravity * (-self.variables["position_d"].value)
        if self.name == "energy_kinetic_rotational" or self.name == "energy_kinetic" or self.name == "energy_total":
            for i, axis in enumerate(["omega_p", "omega_q", "omega_r"]):
                m_i = self.inertia_matrix[i, i]
                val += 1 / 2 * m_i * self.variables[axis].value ** 2
        if self.name == "energy_kinetic_translational" or self.name == "energy_kinetic" or self.name == "energy_total":
            val += 1 / 2 * self.mass * self.variables["Va"].value ** 2

        return val