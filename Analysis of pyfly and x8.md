# Analysis of `pyfly` and `x8`

Here we analyse two codebases, `pyfly` and `x8`. `pyfly` is a python implementation of an aerodynamic model of a small fixed-wing aircraft the _Skywalker X8_, which itself is represented in `x8`.

## `pyfly`

`pyfly` takes in a `x8_param.mat`  file for the aerodynamic model. If we were to use this software, we would require such a file as well.

It also takes in `pyfly_config.json` which would require modification for our purposes, since we are only considering four degrees of freedom instead of six in the code.

A key feature of the `pyfly` codebase is the use of a conventional PID controller tuned to the _Skywalker X8_. Step-wise control outputs, control states and control commands among others can be plotted for analysis. For the purposes of our model, this would prove helpful in the tuning of our PID.

`pid_controller.py` can be modified to find the optimal tune for the PID, if required.

## `x8`

As the name of the repository suggests, it models after the _Skywalker X8_ UAV. It is a comprehensive aerodynamic model of the UAV, with equations based of [this paper](https://folk.ntnu.no/torarnj/icuasX8.pdf).

`forces.m` considers the force equations for the aerodynamic model, and we would utilise these equations for our purposes. It takes in a 3x1 matrix of the air velocity, and considers the change of the coefficient of lift based on the angle of attack $\alpha$. We will look at the longitudinal mode of the code in particular.

A controller is represented in `simX8.m` and will be applicable for our purposes. 

## Link to codebases

[x8](https://github.com/krisgry/x8/blob/master/forces.m)

[pyfly](https://github.com/eivindeb/pyfly)