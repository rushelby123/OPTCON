Overview

	This Python code is a simulation and control framework for a drone system, including tasks related to trajectory planning, tracking performance evaluation, and Model Predictive Control. 		The code is structured into different tasks, each focusing on specific aspects of the drone's behavior and control.

Tasks

-Task 1: Trajectory between Two Positions
	This task generates the optimal trajectory exploiting a Newton's like algorithm following a reference step trajectory between two position equilibira. 

-Task 2a: Smooth Trajectory between Two Positions
	This task is similar to Task 1 but aims to generate a smoother trajectory between two equilibria. It still uses the Newton's-like optimization algorithm, this time to follow a reference 
	sigmoid trajectory.

-Task 2b (Velocity Control): Smooth Trajectory between Two Velocities
	This task generates a smooth trajectory between two velocity equilibria, considering velocity control. It uses a similar approach to Task 2 but focuses on controlling the drone's initial 		and final velocity.

-Task 3: Trajectory Tracking 
	Task 3 evaluates the tracking performance of a Linear Quadratic Regulator (LQR) based on trajectories generated in Task 2. 

-Task 4: Model Predictive Control (MPC)
	Task 4 implements Model Predictive Control (MPC) for regulating the drone's behavior. It includes an MPC solver that predicts the future states and inputs, applying the optimal control at 	each time step.

Running the Code

	To run specific tasks, set the corresponding task flag to True and others to False at the beginning of the code. Adjust parameters and options as needed. Run the entire script to execute 		the selected tasks.

	Note: Each task may have visualization options (visualize_insight). If set to True, the code will generate plots to provide insights into the algorithm's behavior.