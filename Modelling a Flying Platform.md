# Modelling a Flying Platform

We develop the model equations of a small flying platform, given _four_ degrees of freedom. 

## Equations of Motion

The Newtonian equations of motion of a flying platform is given as $^{[1]}$
$$
m\dot v+\omega \cross mv=F
$$

$$
I\dot\omega +\omega\cross Iv=M
$$

where $F$ is the force vector of the system and $M$ is the moment vector of the system.  

### Lift

The lift force can be calculated as
$$
F_L=\frac 12 C_L\rho {v_a}^2A
$$
where $C_L$ is the coefficient of lift, $V_a$ is the airspeed and $A$ is the wing surface.

### Drag

The drag force can be calculated as
$$
F_D=\frac 12 C_D\rho {v_a}^2A
$$
where $C_D$ is the coefficient of drag.

### Pitch Moment

The pitch moment of the flying platform can be given as
$$

$$

## PID

We take in the velocity vector of the air $v_a$, angle of attack $\alpha$, angular velocity vector of the plane $\omega$, and the pitch angle $\theta$.

The control actuators are the throttle, controlling the velocity vector, and the elevons, controlling the pitch.



## References

$^{[1]}$ K. Gryte, R. Hann, M. Alam, J. Rohác, T. A. Johansen, T. I. Fossen, [*Aerodynamic modeling of the Skywalker X8 Fixed-Wing Unmanned Aerial Vehicle*](https://folk.ntnu.no/torarnj/icuasX8.pdf)

$^{[2]}$ Bohn, Eivind and Coates, Erlend M and Moe, Signe and Johansen, Tor Arne [PyFly - Python Fixed Wing Flight Simulator](https://github.com/eivindeb/pyfly)

$^{[3]}$ A Derivation of the Equations of Motion
