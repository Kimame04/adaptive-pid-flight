# A Derivation of the Equations of Motion

We calculate the equations of motion of a flying platform.

## Terminology

$\alpha$ represents the pitch angle of the plane

$\theta$ is the angle-of-attack of the plane

$u$ is the fluid velocity of the plane

$C_D$ is the coefficient of drag of thenplane

$\rho$ is the fluid density (in this case, air)

$A$ is the overall wing area.

$y_0$ is the ideal height (set point)

$y$ is the current height

$F_L$ is the lift force of the plane

## Derivation

In the y-axis, the equation of the motion of a flying platform is given by:
$$
\begin{align}
m\ddot y&=F_{L}\cos\theta -mg+F_T \sin \theta\\
\end{align}
$$
In the x-axis, the equation of motion of a flying platform is given by:
$$
\begin{align}
m\ddot x&=F_T\cos\theta -F_L\sin\theta-\frac 12C_D\rho u^2A
\end{align}
$$

### Lift

Lift, $F_L$ can be calculated as
$$
F_L=\frac {dP_{wing}}{dt}=\frac 12 C_L\rho u^2A
$$
And using the Conservation of Momentum, we obtain for the x-axis,
$$
P_{air,x}=P_{wing,x}+P_{air,x}
$$

### Momentum

Using known equations for the air, we get
$$
\rho Vu=P_{wing,x}+\rho Vu\cos\theta
$$

$$
P_{wing,x}=\rho Vu\cdot(1-\cos\theta)
$$

For the y-axis,
$$
P_{air,y}=P_{wing,y}+P_{air,y}
$$

$$
0=P_{wing,y}+\rho Vu\sin\theta
$$

$$
P_{wing,y}=-\rho Vu\sin\theta P_{wing,y}=-\rho Vu\sin\theta
$$

We can then get the overall momentum of the wing,
$$
\begin{align}
P_{wing}&=\sqrt {{P_{wing,x}}^2+{P_{wing,y}}^2}\\
&=\rho V u \sqrt{(1-\cos\theta)^2+\sin^2\theta}\\
&=\sqrt 2 \rho V u \sqrt{1-\cos\theta}
\end{align}
$$

### Overall

Putting it all together, we get
$$
m\ddot y=-\rho Vu\sin\theta-mg+F_T\sin\theta
$$

$$
m\ddot x= F_T\cos\theta -\rho V u\cdot(1-\cos\theta)-\frac 12 C_D \rho u^2 A
$$

