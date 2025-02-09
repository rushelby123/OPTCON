import numpy as np
import matplotlib.pyplot as plt
import cost as ct
import dynamics as dyn
import matplotlib.animation as animation
from scipy.special import expit

def plot_text(text, x=0.2, y=0.5):
    plt.text(x, y, text)
    plt.axis('off')
    plt.show()

def get_symmetric_reference_curve(x_eq1, u_eq1, x_eq2, u_eq2, TT):
    # Create the state reference matrix 
    x_ref = np.zeros((len(x_eq1),TT))
    x_ref[:,:TT//2] = x_eq1[:, np.newaxis]
    x_ref[:,TT//2:] = x_eq2[:, np.newaxis]

    # Create the input reference matrix
    u_ref = np.zeros((len(u_eq1),TT))
    u_ref[:,:TT//2] = u_eq1[:, np.newaxis]
    u_ref[:,TT//2:] = u_eq2[:, np.newaxis]

    # Print the matrices
    #print("x_ref:\n", x_ref)
    #print("u_ref:\n", u_ref)

    return x_ref, u_ref

def plot_reference_curve(x_ref, u_ref,TT, simulation_time):
    time = np.linspace(0, simulation_time, TT)
    labels = ['x', 'y', 'theta', 'alpha', 'Vx', 'Vy', 'Wa', 'Wo']
    labels_u = ['Fs', 'Fd']
    # Plot x_ref
    for i in range(0, x_ref.shape[0], 2):
        plt.figure(figsize=(10, 6))
        plt.plot(time, x_ref[i, :], label=labels[i], linestyle='--')
        if i+1 < x_ref.shape[0]:  # Check if i+1 is within the number of rows in x_ref
            plt.plot(time, x_ref[i+1, :], label=labels[i+1], linestyle='--')
        plt.title('{} and {} over time'.format(labels[i], labels[i+1]))
        plt.xlabel('Time')
        plt.ylabel('State value')
        plt.legend()
        plt.show()

    # Plot u_ref
    plt.figure(figsize=(10, 6))
    for i in range(u_ref.shape[0]):
        plt.plot(time, u_ref[i, :], label=labels_u[i], linestyle='--')
    plt.title('Fs and Fd over time')
    plt.xlabel('Time')
    plt.ylabel('u_ref')
    plt.legend()
    plt.show()

def plot_trajectory(x_ref, TT, simulation_time):
    time = np.linspace(0, simulation_time, TT)
    labels = ['x', 'y', 'theta', 'alpha', 'Vx', 'Vy', 'Wa', 'Wo']

    # Plot x_ref
    for i in range(0, x_ref.shape[0], 2):
        plt.figure(figsize=(10, 6))
        plt.plot(time, x_ref[i, :], label=labels[i])
        if i+1 < x_ref.shape[0]:  # Check if i+1 is within the number of rows in x_ref
            plt.plot(time, x_ref[i+1, :], label=labels[i+1])
        plt.title('{} and {} over time'.format(labels[i], labels[i+1]))
        plt.xlabel('Time')
        plt.ylabel('State value')
        plt.legend()
        plt.show()

def plot_curves_with_reference(x_ref, u_ref, x_opt, u_opt, TT, simulation_time):
    time = np.linspace(0, simulation_time, TT)
    labels = ['x', 'y', 'theta', 'alpha', 'Vx', 'Vy', 'Wa', 'Wo']
    labels_u = ['Fs', 'Fd']

    # Plot x_ref and x_opt
    for i in range(0, x_ref.shape[0], 2):
        plt.figure(figsize=(10, 6))
        plt.plot(time, x_ref[i, :], label='Reference '+labels[i], linestyle='--', color='blue')
        plt.plot(time, x_opt[i, :], label='Optimal '+labels[i], color='blue')
        if i+1 < x_ref.shape[0]:  # Check if i+1 is within the number of rows in x_ref
            plt.plot(time, x_ref[i+1, :], label='Reference '+labels[i+1], linestyle='--', color='red')
            plt.plot(time, x_opt[i+1, :], label='Optimal '+labels[i+1], color='red')
        plt.title('{} and {} over time'.format(labels[i], labels[i+1]))
        plt.xlabel('Time')
        plt.ylabel('State value')
        plt.legend()
        plt.show()

    # Plot u_ref and u_opt
    plt.figure(figsize=(10, 6))
    for i in range(0, u_ref.shape[0], 2):
        plt.plot(time, u_ref[i, :], label='Reference '+labels_u[i], linestyle='--', color='blue')
        plt.plot(time, u_opt[i, :], label='Optimal '+labels_u[i], color='blue')
        if i+1 < u_ref.shape[0]:  # Check if i+1 is within the number of rows in u_ref
            plt.plot(time, u_ref[i+1, :], label='Reference '+labels_u[i+1], linestyle='--', color='red')
            plt.plot(time, u_opt[i+1, :], label='Optimal '+labels_u[i+1], color='red')
        plt.title('{} and {} over time'.format(labels_u[i], labels_u[i+1]))
        plt.xlabel('Time')
        plt.ylabel('Control value')
        plt.legend()
        plt.show()

def plot_curves_with_reference_and_tracked(x_ref, u_ref, x_opt, u_opt, x_Tracked, u_Tracked,iteration_counter, TT, simulation_time):
    time = np.linspace(0, simulation_time, TT)
    labels = ['x', 'y', 'theta', 'alpha', 'Vx', 'Vy', 'Wa', 'Wo']
    labels_u = ['Fs', 'Fd']

    # Plot x_ref, x_opt and x_Tracked
    for i in range(0, x_ref.shape[0], 2):
        plt.figure(figsize=(10, 6))
        plt.plot(time, x_ref[i, :], label='Reference '+labels[i], linestyle='--', color='blue')
        plt.plot(time, x_opt[i, :], label='Optimal '+labels[i], linestyle=':', color='blue')
        plt.plot(time, x_Tracked[i, :], label='Tracked '+labels[i], color='blue')
        if i+1 < x_ref.shape[0]:  # Check if i+1 is within the number of rows in x_ref
            plt.plot(time, x_ref[i+1, :], label='Reference '+labels[i+1], linestyle='--', color='red')
            plt.plot(time, x_opt[i+1, :], label='Optimal '+labels[i+1], linestyle=':', color='red')
            plt.plot(time, x_Tracked[i+1, :], label='Tracked '+labels[i+1], color='red')
        plt.title('{}-th iteration: {} and {} over time'.format(iteration_counter, labels[i], labels[i+1]))
        plt.xlabel('Time')
        plt.ylabel('State value')
        plt.legend()
        plt.show()

    # Plot u_ref, u_opt and u_Tracked
    plt.figure(figsize=(10, 6))
    for i in range(0, u_ref.shape[0], 2):
        plt.plot(time, u_ref[i, :], label='Reference '+labels_u[i], linestyle='--', color='blue')
        plt.plot(time, u_opt[i, :], label='Optimal '+labels_u[i], linestyle=':', color='blue')
        plt.plot(time, u_Tracked[i, :], label='Tracked '+labels_u[i], color='blue')
        if i+1 < u_ref.shape[0]:  # Check if i+1 is within the number of rows in u_ref
            plt.plot(time, u_ref[i+1, :], label='Reference '+labels_u[i+1], linestyle='--', color='red')
            plt.plot(time, u_opt[i+1, :], label='Optimal '+labels_u[i+1], linestyle=':', color='red')
            plt.plot(time, u_Tracked[i+1, :], label='Tracked '+labels_u[i+1], color='red')
        plt.title('{}-th iteration: {} and {} over time'.format(iteration_counter, labels_u[i], labels_u[i+1]))
        plt.xlabel('Time')
        plt.ylabel('Control value')
        plt.legend()
        plt.show()

def plot_reference_real_mpc(x_ref, u_ref, x_real, u_real, x_mpc, u_mpc,iteration_counter, TT, simulation_time):
    time = np.linspace(0, simulation_time, TT)
    labels = ['x', 'y', 'theta', 'alpha', 'Vx', 'Vy', 'Wa', 'Wo']
    labels_u = ['Fs', 'Fd']

    # Plot x_ref, x_real and x_mpc
    for i in range(0, x_ref.shape[0], 2):
        plt.figure(figsize=(10, 6))
        plt.plot(time, x_ref[i, :], label='Reference '+labels[i], linestyle='--', color='blue')
        plt.plot(time, x_real[i, :], label='Real '+labels[i], linestyle=':', color='blue')
        plt.plot(time, x_mpc[i, :], label='MPC '+labels[i], color='blue')
        if i+1 < x_ref.shape[0]:  # Check if i+1 is within the number of rows in x_ref
            plt.plot(time, x_ref[i+1, :], label='Reference '+labels[i+1], linestyle='--', color='red')
            plt.plot(time, x_real[i+1, :], label='Real '+labels[i+1], linestyle=':', color='red')
            plt.plot(time, x_mpc[i+1, :], label='MPC '+labels[i+1], color='red')
        plt.title('{}-th iteration: {} and {} over time'.format(iteration_counter, labels[i], labels[i+1]))
        plt.xlabel('Time')
        plt.ylabel('State value')
        plt.legend()
        plt.show()

    # Plot u_ref, u_real and u_mpc
    plt.figure(figsize=(10, 6))
    for i in range(0, u_ref.shape[0], 2):
        plt.plot(time, u_ref[i, :], label='Reference '+labels_u[i], linestyle='--', color='blue')
        plt.plot(time, u_real[i, :], label='realimal '+labels_u[i], linestyle=':', color='blue')
        plt.plot(time, u_mpc[i, :], label='mpc '+labels_u[i], color='blue')
        if i+1 < u_ref.shape[0]:  # Check if i+1 is within the number of rows in u_ref
            plt.plot(time, u_ref[i+1, :], label='Reference '+labels_u[i+1], linestyle='--', color='red')
            plt.plot(time, u_real[i+1, :], label='realimal '+labels_u[i+1], linestyle=':', color='red')
            plt.plot(time, u_mpc[i+1, :], label='mpc '+labels_u[i+1], color='red')
        plt.title('{}-th iteration: {} and {} over time'.format(iteration_counter, labels[i], labels[i+1]))
        plt.xlabel('Time')
        plt.ylabel('Control value')
        plt.legend()
        plt.show()



def generate_sigmoid_trajectory(TT):
    time = np.linspace(0, 1, TT)
    return expit(15 * (time - 0.5))

def get_smooth_reference_curve(x_eq1, u_eq1, x_eq2, u_eq2, TT):

    smooth_trajectory = generate_sigmoid_trajectory(TT)
    
    x_ref = np.zeros((len(x_eq1), TT))
    x_eq1=np.tile(x_eq1, ( TT , 1))
    x_eq2=np.tile(x_eq2, ( TT , 1))  

    u_ref = np.zeros((len(u_eq1), TT))
    u_eq1=np.tile(u_eq1, ( TT , 1))
    u_eq2=np.tile(u_eq2, ( TT , 1)) 

    for i in range(8):
        x_ref[i, :] = x_eq1[:, i] + (x_eq2[:, i] - x_eq1[:, i]) * smooth_trajectory

    for i in range(2):  
        u_ref[i, :] = u_eq1[:, i] + (u_eq2[:, i] - u_eq1[:, i]) * smooth_trajectory
    
    return x_ref, u_ref



def plot_drone(x, u, TT_max, simulation_time, target_point):
    fig, ax = plt.subplots()

    # Define the length of the drone and the radius of the helices
    drone_length = dyn.ll_drone*5
    helix_radius = drone_length/4
    pendulum_length = dyn.ll_load

    # Plot the curve defined by x[0, tt] and x[1, tt] ##### to see the trajectory followed by the drone
    ax.plot(x[0, :], x[1, :], 'aquamarine')  # Change the color of the curve to light blue

    # Initialize the drone and helices lines
    drone_line, = ax.plot([], [], 'k-')  # Change the color of the drone to black
    helix1_line, = ax.plot([], [], 'k-')  # Change the color of the helices to black
    helix2_line, = ax.plot([], [], 'k-')  # Change the color of the helices to black
    pendulum_line, = ax.plot([], [], 'darkgoldenrod')  # Change the color of the pendulum to yellow

    # Initialize the red dot at the center of the drone
    center_dot = ax.scatter([], [], color='r')
    pendulum_end = ax.scatter([], [], color='g')

    # Initialize the time text
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    # Initialize the red dot at the center of the drone
    center_dot = ax.scatter([], [], color='r')

    # Initialize the target point
    ax.scatter(target_point[0],target_point[1], marker='*', color='yellow')

    # Initialize the thrust vectors
    thrust1 = ax.quiver(x[0, 0], x[1, 0], 0, 0, angles='xy', scale_units='xy', scale=1, color='green')
    thrust2 = ax.quiver(x[0, 0], x[1, 0], 0, 0, angles='xy', scale_units='xy', scale=1, color='green')

    # Initialize the velocity texts
    Vx_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
    Vy_text = ax.text(0.02, 0.80, '', transform=ax.transAxes)
    Wa_text = ax.text(0.02, 0.75, '', transform=ax.transAxes)
    Wo_text = ax.text(0.02, 0.70, '', transform=ax.transAxes)


    ax.set_xlim(-1.5, 1.5)  # Set x limits
    ax.set_ylim(-1.5, 1.5)  # Set y limits
    ax.grid(True) # Show the grid

# Update function for the animation
    def update(tt):
        # Calculate the coordinates of the drone and pendulum line and the helices
        drone_x = [x[0, tt] - drone_length / 2, x[0, tt] + drone_length / 2]
        drone_y = [x[1, tt], x[1, tt]]
        helix1_x = drone_x[0] + helix_radius * np.cos(np.linspace(0, 2*np.pi, 100))
        helix1_y = drone_y[0] + helix_radius * np.sin(np.linspace(0, 2*np.pi, 100))
        helix2_x = drone_x[1] + helix_radius * np.cos(np.linspace(0, 2*np.pi, 100))
        helix2_y = drone_y[1] + helix_radius * np.sin(np.linspace(0, 2*np.pi, 100))
        pendulum_x = x[0, tt] + pendulum_length * np.sin(x[2, tt])
        pendulum_y = x[1, tt] - pendulum_length * np.cos(x[2, tt])

        # Update the pendulum line
        pendulum_line.set_data([x[0, tt], pendulum_x], [x[1, tt], pendulum_y])

        # Adjust the coordinates so that the center of the drone is at the origin
        drone_x, drone_y = [coord - x[0, tt] for coord in drone_x], [coord - x[1, tt] for coord in drone_y]
        helix1_x, helix1_y = [coord - x[0, tt] for coord in helix1_x], [coord - x[1, tt] for coord in helix1_y]
        helix2_x, helix2_y = [coord - x[0, tt] for coord in helix2_x], [coord - x[1, tt] for coord in helix2_y]

        # Calculate the rotation matrix
        c, s = np.cos(x[3, tt]), np.sin(x[3, tt])
        R = np.array(((c, -s), (s, c)))

        # Apply the rotation matrix to the drone and helices lines
        drone_x, drone_y = R @ [drone_x, drone_y]
        helix1_x, helix1_y = R @ [helix1_x, helix1_y]
        helix2_x, helix2_y = R @ [helix2_x, helix2_y]

        # Adjust the coordinates back
        drone_x, drone_y = [coord + x[0, tt] for coord in drone_x], [coord + x[1, tt] for coord in drone_y]
        helix1_x, helix1_y = [coord + x[0, tt] for coord in helix1_x], [coord + x[1, tt] for coord in helix1_y]
        helix2_x, helix2_y = [coord + x[0, tt] for coord in helix2_x], [coord + x[1, tt] for coord in helix2_y]

        # Update the drone and helices lines
        drone_line.set_data(drone_x, drone_y)
        helix1_line.set_data(helix1_x, helix1_y)
        helix2_line.set_data(helix2_x, helix2_y)

        # Update the thrust vectors
        thrust1.set_UVC(-u[0, tt]*np.sin(x[3, tt]), u[0, tt]*np.cos(x[3, tt]))
        thrust1.set_offsets([drone_x[0], drone_y[0]])  # left end point of the drone
        thrust2.set_UVC(-u[1, tt]*np.sin(x[3, tt]), u[1, tt]*np.cos(x[3, tt]))
        thrust2.set_offsets([drone_x[-1], drone_y[-1]])  # right end point of the drone

        # Update the red dot at the center of the drone
        center_dot.set_offsets([x[0, tt], x[1, tt]])
        pendulum_end.set_offsets([pendulum_x, pendulum_y])

        # Update the velocity texts
        Vx_text.set_text('Vx = %.2f' % x[4, tt])
        Vy_text.set_text('Vy = %.2f' % x[5, tt])
        Wa_text.set_text('Wa = %.2f' % x[6, tt])
        Wo_text.set_text('Wo = %.2f' % x[7, tt])


        # Update the limits of the axes to center the plot on the center_dot
        #ax.set_xlim(x[0, tt] - 1.5, x[0, tt] + 1.5)
        #ax.set_ylim(x[1, tt] - 1.5, x[1, tt] + 1.5)

        # Update the time text
        time_text.set_text('Time = %.1f' % (tt * simulation_time / TT_max))

        return drone_line, helix1_line, helix2_line, center_dot, pendulum_line, pendulum_end, time_text, thrust1, thrust2, Vx_text, Vy_text, Wa_text, Wo_text,
    
    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=TT_max, interval=simulation_time/TT_max*1000, blit=True)

    plt.show()

def plot_tracking_performance(x_ref, u_ref, x_opt, u_opt, x_Tracked, u_Tracked, TT, simulation_time):
    time = np.linspace(0, simulation_time, TT)
    labels = ['x', 'y', 'theta', 'alpha', 'Vx', 'Vy', 'Wa', 'Wo']
    labels_u = ['Fs', 'Fd']

    # Plot x_ref, x_opt and x_Tracked
    for i in range(0, x_ref.shape[0], 2):
        plt.figure(figsize=(10, 6))
        plt.plot(time, x_ref[i, :], label='Reference '+labels[i], linestyle='--', color='blue')
        plt.plot(time, x_opt[i, :], label='Optimal '+labels[i], linestyle=':', color='blue')
        plt.plot(time, x_Tracked[i, :], label='Tracked '+labels[i], color='blue')
        if i+1 < x_ref.shape[0]:  # Check if i+1 is within the number of rows in x_ref
            plt.plot(time, x_ref[i+1, :], label='Reference '+labels[i+1], linestyle='--', color='red')
            plt.plot(time, x_opt[i+1, :], label='Optimal '+labels[i+1], linestyle=':', color='red')
            plt.plot(time, x_Tracked[i+1, :], label='Tracked '+labels[i+1], color='red')
        plt.title('Tracking performace: {} and {} over time'.format( labels[i], labels[i+1]))
        plt.xlabel('Time')
        plt.ylabel('State value')
        plt.legend()
        plt.show()

    # Plot u_ref, u_opt and u_Tracked
    plt.figure(figsize=(10, 6))
    for i in range(0, u_ref.shape[0], 2):
        plt.plot(time, u_ref[i, :], label='Reference '+labels_u[i], linestyle='--', color='blue')
        plt.plot(time, u_opt[i, :], label='Optimal '+labels_u[i], linestyle=':', color='blue')
        plt.plot(time, u_Tracked[i, :], label='Tracked '+labels_u[i], color='blue')
        if i+1 < u_ref.shape[0]:  # Check if i+1 is within the number of rows in u_ref
            plt.plot(time, u_ref[i+1, :], label='Reference '+labels_u[i+1], linestyle='--', color='red')
            plt.plot(time, u_opt[i+1, :], label='Optimal '+labels_u[i+1], linestyle=':', color='red')
            plt.plot(time, u_Tracked[i+1, :], label='Tracked '+labels_u[i+1], color='red')
        plt.title('Tracking performace: {} and {} over time'.format( labels_u[i], labels_u[i+1]))
        plt.xlabel('Time')
        plt.ylabel('Control value')
        plt.legend()
        plt.show()

"""
def plot_mpc(T_sim, ns, ni, xx_real_mpc, x_T3opt, uu_real_mpc, u_T3opt): 

    time = np.arange(T_sim)
    labels = ['x', 'y', 'theta', 'alpha', 'Vx', 'Vy', 'Wa', 'Wo']
    labels_u = ['Fs', 'Fd']

    for i in range(ns):
        fig, ax = pyplot.subplots()
        ax.plot(time, xx_real_mpc[i,:], linewidth=2)
        ax.plot(time, x_T3opt[i,:], '--r', linewidth=2)  
        ax.grid()
        ax.set_ylabel(f'$x_{i+1}$')

        if 1 or np.amax(x_T3opt[i,:]) > 100:
            ax.set_ylim([-5,5])

        ax.set_xlim([-1,T_sim])
        ax.legend(['MPC', 'LQR'])
        plt.title('{} over time'.format(labels[i]))

    for i in range(ni):
        fig, ax = pyplot.subplots()
        ax.plot(time, uu_real_mpc[i,:], linewidth=2)
        ax.plot(time, u_T3opt[i,:], '--r', linewidth=2)  
        ax.grid()
        ax.set_ylabel(f'$u_{i+1}$')

        if 1 or np.amax(u_T3opt[i,:]) > 100:
            ax.set_ylim([-5,5])

        ax.set_xlim([-1,T_sim])
        ax.legend(['MPC', 'LQR'])
        plt.title('{} over time'.format(labels_u[i]))

    plt.show()

"""
def plot_mpc(T_sim, ns, ni, xx_real_mpc, x_T3opt, uu_real_mpc, u_T3opt): 

    time = np.arange(T_sim)
    labels = ['x', 'y', 'theta', 'alpha', 'Vx', 'Vy', 'Wa', 'Wo']
    labels_u = ['Fs', 'Fd']

    for i in range(ns):
        fig, ax = plt.subplots()
        ax.plot(time, xx_real_mpc[i,:], linewidth=2)
        ax.plot(time, x_T3opt[i,:], '--r', linewidth=2)  
        ax.grid()
        ax.set_ylabel(f'$x_{i+1}$')

        if 1 or np.amax(x_T3opt[i,:]) > 100:
            ax.set_ylim([-5,5])

        ax.set_xlim([-1,T_sim])
        ax.legend(['MPC', 'LQR'])
        plt.title('{} over time'.format(labels[i]))
        plt.show()

    for i in range(ni):
        fig, ax = plt.subplots()
        ax.plot(time, uu_real_mpc[i,:], linewidth=2)
        ax.plot(time, u_T3opt[i,:], '--r', linewidth=2)  
        ax.grid()
        ax.set_ylabel(f'$u_{i+1}$')

        if 1 or np.amax(u_T3opt[i,:]) > 100:
            ax.set_ylim([-5,5])

        ax.set_xlim([-1,T_sim])
        ax.legend(['MPC', 'LQR'])
        plt.title('{} over time'.format(labels_u[i]))
        plt.show()
    

def plot_descent_and_cost(Descent, JJ, i):
    
    indices = range(i)
    plt.figure()
    
    # Plot the norm of the descent direction along iterations
    plt.title("Norm of the descent direction along iterations ")
    plt.semilogy(indices,abs(Descent[1:i+1]), label='Norm of descent direction')
    plt.xlabel('Number of iterations')
    plt.ylabel('||$\\nabla J(\\mathbf{u}^k)||$')
    plt.grid()
    plt.show()
    


    # Plot the cost along iterations up to index i
    plt.figure()
    plt.title("Cost along iterations")
    plt.semilogy(indices,JJ[1:i+1], label='Cost')
    plt.xlabel('Number of iterations')
    plt.ylabel('$J(\\mathbf{u}^k)$')
    plt.grid()
    plt.show()

"""
def plot_tracking_error(initial_condition,x_real_mpc, T_sim, T_pred, x_T3opt):
    tracking_error = np.linalg.norm(x_real_mpc[:, :T_sim - T_pred] - x_T3opt[:, :T_sim - T_pred], axis=0)

    plt.plot(range(T_sim - T_pred), tracking_error, label=f'Initial Condition: {initial_condition}')
    plt.xlabel('Time Step')
    plt.ylabel('Tracking Error')
    plt.title('Tracking Error for Different Initial Conditions')
    plt.legend()
    plt.show()

"""

def plot_tracking_error(initial_condition, x_real_mpc,u_real_mpc, x_T3opt,u_T3opt, ns,ni, T_sim, T_pred):
 
    time = np.arange(T_sim)
    labels = ['x', 'y', 'theta', 'alpha', 'Vx', 'Vy', 'Wa', 'Wo']
    labels_u = ['Fs', 'Fd']
 
    for i in range(ns):
        fig, ax = plt.subplots()
        ax.plot(time, x_real_mpc[i,:]- x_T3opt[i,:], linewidth=2)
        ax.grid()
        ax.set_ylabel(f'$x_{i+1}$')
 
        ax.set_xlim([-1,T_sim])
        plt.title('Error: {} over time'.format(labels[i]))
        plt.show()

    for i in range(ni):
        fig, ax = plt.subplots()
        ax.plot(time, u_real_mpc[i,:]-u_T3opt[i,:], linewidth=2)
        ax.grid()

        ax.set_ylabel(f'$u_{i+1}$')
 
        ax.set_ylim([-5,5])
 
        ax.set_xlim([-1,T_sim])
        plt.title('Error: {} over time'.format(labels_u[i]))
        plt.show()
