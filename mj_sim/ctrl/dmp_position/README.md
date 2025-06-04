
# DMP Position Controller

The DMP (Dynamic Movement Primitive) Position Controller is a method used for generating trajectories for robotic systems based on DMPs. Let's delve into the mathematical concepts involved:

### 1. Dynamic Movement Primitives (DMPs)

Dynamic Movement Primitives (DMPs) are a framework for learning and reproducing complex, rhythmic movements in robotics. They are represented by a non-linear differential equation:

$$
\tau \dot{v} = K(v_g - y) - Dv + F_{\text{target}}
$$

where:
- $\tau$ is the movement duration scaling factor,
- $y$ is the current position,
- $v$ is the velocity,
- $v_g$ is the goal velocity,
- $K$ and $D$ are gain matrices,
- $F_{\text{target}}$ is the forcing term.

### 2. Canonical System

The Canonical System (CS) governs the timing of the movement. It is represented by a first-order differential equation:

$$
\tau \dot{x} = -\alpha_x x
$$

where $x$ is the phase variable and $\alpha_x$ is a constant.

### 3. Basis Functions

The trajectory in a DMP is represented as a weighted sum of Gaussian basis functions:

$$
f(x) = \frac{\sum_{j=1}^{N} \psi_j w_j}{\sum_{j=1}^{N} \psi_j}
$$

where $\psi_j = \exp(-h_j(x - c_j)^2)$ are the Gaussian basis functions, $c_j$ are the centers, $h_j$ are the variances, and $w_j$ are the weights.

### 4. Forcing Term

The forcing term is responsible for driving the system along the desired trajectory. It is computed as:

$$
F_{\text{target}} = \frac{\sum_{j=1}^{N} x \psi_j w_j}{\sum_{j=1}^{N} \psi_j}
$$

### 5. System Dynamics

The dynamics of the DMP system are described by the second-order differential equation:

$$
\tau^2 \ddot{y} = K(v_g - y) - D \dot{y} + F_{\text{target}}
$$

The acceleration $\ddot{y}$ is integrated numerically to obtain velocity and position.

### 6. Training

To train the DMP, demonstrations of the desired trajectory are provided. The weights $w_j$ are learned by solving the following system of equations:

$$
A w = f
$$

where $A$ is the matrix of basis functions evaluated at each time step, and $f$ is the desired trajectory.

### 7. Goal Change Parameters

When the desired goal changes during execution, the controller must adapt smoothly. This involves updating the goal position and possibly the orientation to transition smoothly to the new goal.

### 8. Scaling Factor

The scaling factor adjusts the magnitude of the trajectory based on the difference between the initial and goal positions:

$$
D_p = \text{diag}(g - y_0)
$$

### Acknowledgements
Implementation is based on the work `simple_dmp` by [Iñigo Iturrate](https://portal.findresearcher.sdu.dk/da/persons/i%C3%B1igo-iturrate)
