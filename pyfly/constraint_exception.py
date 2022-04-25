class ConstraintException(Exception):
    def __init__(self, variable, value, limit):
        self.message = "Constraint on {} violated ({}/{})".format(variable, value, limit)
        self.variable = variable
