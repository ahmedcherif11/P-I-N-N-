# Predicting Shuttlecock Trajectory with Physics-Informed Neural Networks (PINN)
This repository implements a Physics-Informed Neural Network (PINN) to model and predict the trajectory of a shuttlecock in 2D space. The project highlights the strengths of integrating physical laws into deep learning models, enabling precise predictions even with limited data.


The model is designed to:
1. Predict shuttlecock trajectories based on physical principles (e.g., gravity, drag).
2. Evaluate the benefits of training on sparse data (e.g., 20 representative points) while maintaining high accuracy.

---
## Project Overview
### Objectives :
- Comprehensive Modeling: Develop a PINN to predict a shuttlecock's trajectory using a combination of physical equations and neural network training.
- Data Efficiency: Demonstrate the advantage of PINNs by training with sparse data (20 points) compared to training on the full dataset.
- Robust Prediction: Mitigate the effects of noise in trajectory data through physics-based constraints.


---
## Dataset Generation

The dataset simulates the 2D motion of a shuttlecock under the influence of gravity and drag. 

### Simulation Parameters:
- **Gravity (g)**: 9.81 m/s²
- **Drag Coefficient (C_d)**: 0.5
- **Cross-sectional Area (A)**: 0.01 m²
- **Mass (m)**: 0.02 kg
- **Initial Velocity (v₀)**: User-defined (e.g., 30 m/s)
- **Launch Angle (θ)**: User-defined (e.g., 45°)

A differential equation governs the shuttlecock's motion:
- **Horizontal Acceleration**:$a_x = -C_d \cdot v \cdot v_x$ 
- **Vertical Acceleration**: $a_y = -g - C_d \cdot v \cdot v_y$
  Where:  
- $v = \sqrt{v_x^2 + v_y^2}$ is the velocity magnitude  

**Generated Data**: The simulation records $(x, y, v_x, v_y)$ for every time step $t$ by solving these equations iteratively. The shuttlecock's motion stops when its vertical position $y$ becomes negative.
```python
def generate_trajectory_data(v_0, theta, dt=0.01, t_max=10):
    theta_rad = np.radians(theta)  # Convert angle to radians
    v_x = v_0 * np.cos(theta_rad)  # Initial horizontal velocity
    v_y = v_0 * np.sin(theta_rad)  # Initial vertical velocity
    x, y = 0, 0  # Initial position (starting at origin)
    data = []
    time = []  # To store the actual time steps

    for t in np.arange(0, t_max, dt):
        data.append([x, y, v_x, v_y])  # Append the current position and velocity
        time.append(t)  # Append the current time
        v = np.sqrt(v_x**2 + v_y**2)  # Current speed (magnitude of the velocity vector)

        # Air resistance (drag) affecting horizontal velocity (v_x)
        v_x -= (C_d * A / m) * v * v_x * dt  # Update horizontal velocity with drag force

        # Air resistance (drag) and gravity affecting vertical velocity (v_y)
        v_y -= (g + (C_d * A / m) * v * v_y) * dt  # Update vertical velocity with drag and gravity

        # Update positions based on the velocities
        x += v_x * dt  # Update horizontal position
        y += v_y * dt  # Update vertical position

        if y < 0:  # Stop the simulation if the shuttlecock hits the ground (y < 0)
            break

    return np.array(data), np.array(time)  # Return the trajectory data and actual timedef generate_trajectory_data(v_0, theta, dt=0.01, t_max=10):
    theta_rad = np.radians(theta)  # Convert angle to radians
    v_x = v_0 * np.cos(theta_rad)  # Initial horizontal velocity
    v_y = v_0 * np.sin(theta_rad)  # Initial vertical velocity
    x, y = 0, 0  # Initial position (starting at origin)
    data = []
    time = []  # To store the actual time steps

    for t in np.arange(0, t_max, dt):
        data.append([x, y, v_x, v_y])  # Append the current position and velocity
        time.append(t)  # Append the current time
        v = np.sqrt(v_x**2 + v_y**2)  # Current speed (magnitude of the velocity vector)

        # Air resistance (drag) affecting horizontal velocity (v_x)
        v_x -= (C_d * A / m) * v * v_x * dt  # Update horizontal velocity with drag force

        # Air resistance (drag) and gravity affecting vertical velocity (v_y)
        v_y -= (g + (C_d * A / m) * v * v_y) * dt  # Update vertical velocity with drag and gravity

        # Update positions based on the velocities
        x += v_x * dt  # Update horizontal position
        y += v_y * dt  # Update vertical position

        if y < 0:  # Stop the simulation if the shuttlecock hits the ground (y < 0)
            break

    return np.array(data), np.array(time)  # Return the trajectory data and actual time
```

![generated trajectory ](images/Shuttlecock-2D-Trajectory-with-Air-Resistance.png)


---
## Architecture Design

The PINN architecture combines:
1. **Data-Driven Learning**: A neural network approximates the trajectory of the shuttlecock.
2. **Physics-Informed Constraints**: Physical equations serve as regularization terms, ensuring predictions comply with Newtonian mechanics.
   
### Model Components
- **Input**: Time ($t$)
- **Output**:  $[x, y, v_x, v_y]$
- **Structure**:
  - 3 fully connected hidden layers with 256 neurons each
  - Activation: Tanh function
  - Output: 4 neurons representing the trajectory state

#### Loss Function
1. **Data Loss**:
   Measures the difference between predicted ($\hat{y}$) and actual ($y$) trajectory points:
   $\text{Data Loss} = \frac{1}{N} \sum_{i=1}^N \|\hat{y}_i - y_i\|^2$

2. **Physics Loss**:
   Ensures compliance with Newtonian mechanics:
   $Physics Loss= \| a_x - f_x \|^2 + \| a_y - f_y \|^2$
   where:
   - $a_x, a_y$: Predicted accelerations (second derivatives computed via autograd).
   - $f_x, f_y$: Forces acting on the shuttlecock:
     $f_x = -C_d \cdot v \cdot v_x, \quad f_y = -g - C_d \cdot v \cdot v_y$
   - $C_d$: Drag coefficient, $g$: Gravity, $v$: Speed magnitude.

3. **Total Loss**:
   Combines data and physics losses with a regularization term ($\lambda_\text{reg}$):
   $\text{Total Loss} = \text{Data Loss} + \lambda_\text{reg} \cdot \text{Physics Loss}$
```python
# Define PiNN model
class ShuttlecockPiNN(nn.Module):
    def __init__(self):
        super(ShuttlecockPiNN, self).__init__()
        self.hidden = nn.Sequential(
            nn.Linear(1, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 256), nn.Tanh(),
            nn.Linear(256, 4)  # Outputs: x, y, v_x, v_y
        )
        self.C = 0.01  # Drag coefficient
    def forward(self, t):
        return self.hidden(t)
    def compute_ux(self, x_in):
      return torch.autograd.functional.jacobian(self, x_in, create_graph=True)

    def compute_physics_loss(self, t, outputs):
        x, y, v_x, v_y = outputs[:, 0], outputs[:, 1], outputs[:, 2], outputs[:, 3]

        # Compute derivatives using autograd
        v_pred = torch.sqrt(v_x**2 + v_y**2)  # Speed magnitude
        ax = torch.autograd.grad(v_x.sum(), t, create_graph=True)[0]
        ay = torch.autograd.grad(v_y.sum(), t, create_graph=True)[0]

        # Physics constraints: Drag and gravity forces
        fx = -self.C * v_pred * v_x
        fy = -g - self.C * v_pred * v_y

        # Physics loss (difference between predicted and calculated physics)
        physics_loss = torch.mean((ax - fx) ** 2 + (ay - fy) ** 2)
        return physics_loss

    def compute_total_loss(self, t, outputs, targets, lambda_reg=1.0):
        data_loss = torch.mean((outputs - targets) ** 2)
        physics_loss = self.compute_physics_loss(t, outputs)
        return data_loss + lambda_reg * physics_loss

# Instantiate the model
model = ShuttlecockPiNN()
```

#### Key Methods
1. **`forward`**: Computes $x$, $y$, $v_x$, $v_y$ from the input time `t`.
2. **`compute_physics_loss`**:
   - Enforces physical laws (drag and gravity).
   - Penalizes deviations between predicted accelerations ($a_x$, $a_y$) and physical forces.
3. **`compute_total_loss`**:
   - Combines:
     - **Data Loss:** Prediction vs. target error.
     - **Physics Loss:** Consistency with physical equations.
   - Balances both via a regularization parameter ($\lambda$).

### Integration of Physical Loss
- **Design Rationale**:
  - The physical loss enforces compliance with real-world dynamics, reducing overfitting to noisy or limited data.
  - The inclusion of physical laws ensures the model generalizes well, even when trained on a sparse subset of data points.
- **Noise Mitigation**:
  - By regularizing predictions with physical equations, the model ensures smooth and realistic trajectories.
---

## Training Configurations
The PiNN was trained under the following configurations:
1. Training on the **Full Dataset**.
2. Training on only **20 Representative Points**.
---

## Qualitative Results
The following results were observed:
1. **Full Dataset**:  
   - The predicted trajectory closely matched the ground truth.
   - Physics consistency was well-maintained across all time steps.

2. **20 Points Dataset**:  
   - Despite using fewer points, the PiNN effectively predicted the trajectory due to its ability to enforce physical laws.
   - Noise resilience was observed, with the model smoothly reconstructing the trajectory.
![Shuttlecock Trajectory](images/loss-curve-(20-points).png)


### Quantitative Results
| Metric            | Full Dataset | 20 Points Dataset |
|--------------------|--------------|-------------------|
| **Final Loss**     |4.82    |3.55           |
| **Detection Accuracy** | 100%       | 97.63%             |
| **False Positives (FP)** | 0           | 2                 |
| **False Negatives (FN)** | 0           | 2                 |
| **Inference Latency** | 0.0026 seconds | 0.0033 seconds       |
### Visualization
The true trajectory (red dotted line) and the PiNN predictions (blue solid line) for both datasets are visualized below. Training points are shown in green.  
### Training on the **Full Dataset** :
![Shuttlecock Trajectory](images/True-vs-Predicted-Shuttlecock-Trajectory-(all-points).png)
### Training on only **20 Representative Points** :
![Shuttlecock Trajectory](images/True-vs-Predicted-Shuttlecock-Trajectory-(20-points).png)
---
---

## How to Run

### Prerequisites
- Python 3.8+
- PyTorch
- Matplotlib
- Numpy

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/ahmedcherif11/Shuttlecock-Trajectory-Prediction-Using-Physics-Informed-Neural-Networks-PINNs.git
   ```
2. Install dependencies:
3.Open the Notebook: Open the PINN.ipynb file using Jupyter Notebook or google colab:
```bash
jupyter notebook PINN.ipynb
```
4.Modify Simulation Parameters (Optional): You can adjust the initial velocity and angle of projection to generate new data. For example:
```python
# Modify these parameters in the notebook
data = generate_trajectory_data(v_0=40, theta=60)  # Initial velocity 40 m/s, angle 60 degrees
```
- v_0: Initial velocity of the shuttlecock in meters per second (default: 30 m/s).
- theta: Angle of projection in degrees (default: 45°).
5. Run the Notebook

- **Generate Data**: Simulate the shuttlecock's trajectory using physical parameters.  
- **Train on All Data**: Train the PiNN with the full dataset.  
- **Train on 20 Points**: Test the PiNN's efficiency with only 20 representative points.( you can change the number of points) 
- **Visualize Results**: Compare true and predicted trajectories, analyze loss curves, and evaluate accuracy.  


## Conclusion
This project highlights the potential of Physics-Informed Neural Networks to predict the trajectory of objects governed by physical laws. Key takeaways include:
- PiNNs can achieve high accuracy even with limited data by leveraging physical constraints.
- The physics loss significantly improves model robustness to noise and sparse training data.
- This approach can be generalized to other physical systems for trajectory prediction.

---
