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

![generated trajectory ](images/Shuttlecock-2D-Trajectory-with-Air-Resistance.png)


---
