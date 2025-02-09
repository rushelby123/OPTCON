import numpy as np
import dynamics as dyn
import math 
import plots as plt 
import solver as slv 
import cost as ct

#---------------------------------------------------------------------------------
#TASK 1 trajectory between two positions. (The generated trajectory is not smooth)
#---------------------------------------------------------------------------------
if True:
    plt.plot_text("TASK 1 trajectory between two positions.")
    print("TASK 1 trajectory between two positions.")
    simulation_time = 5 #seconds
    visualize_insight = False #if you want to see the plots of the algorithms set it to True
    ct.set_consider_velocity_control(False)

    # initialize variables AS THE INITIAL GUESS FOR THE EQUILIBRIUM POINTS SELECT THE CLOSE FORM SOL
    uu_guess_eq1 = np.array([0.66708 , 0.0])
    uu_guess_eq2 = np.array([0.943393 , 0.0])
    TT=math.ceil(simulation_time/dyn.dt)  

    # initial and final equilibrium points
    x_eq1=np.array([0,0,0,0,0,0,0,0])
    x_eq2=np.array([1,-1,0,0,0,0,0,0])

    # Compute the input for the initial equilibrium point
    u_eq1 = dyn.find_equilibrium_input_xy(uu_guess_eq1, x_eq1[0] , x_eq1[1])
    u_eq2 = dyn.find_equilibrium_input_xy(uu_guess_eq2, x_eq2[0] , x_eq2[1])

    #check if the desired state is an equilibrium point using the definition of equilibrium trjectory
    do_you_want_to_check = False #?
    if do_you_want_to_check:
        xx = np.zeros((dyn.ns,TT))
        uu = np.zeros((dyn.ni,TT))   
        uu = np.tile(u_eq1, (1, uu.shape[1]))
        flag = False
        for tt in range(TT-1):
            xx[:,tt+1] = dyn.dynamics(xx[:,tt], u_eq1)[0]
            if np.all(xx[:,tt+1]-xx[:,tt] != 0):
                flag = True
                print("The equilibrium point is not correct")
                break
        if flag == False:
            print("The equilibrium point is correct")
        plt.plot_trajectory(xx, TT, simulation_time)  

    # Define the reference curve
    x_ref, u_ref = plt.get_symmetric_reference_curve(x_eq1, u_eq1, x_eq2, u_eq2, TT)

    # Plot the reference curve  
    #if visualize_insight:
        #plt.plot_reference_curve(x_ref, u_ref,TT, simulation_time)

    # Implement the Newton's-like algorithm
    max_iters = 1000
    armijo_max_iters = 30
    cc=0.7
    beta=0.7
    initial_step_size=0.5
    term_cond = 10**-4
    x_opt,u_opt,_ = slv.newtons_like_algorithm(x_ref, u_ref, max_iters, armijo_max_iters,visualize_insight,term_cond,beta,cc,initial_step_size) 

    #show animation
    u_transformed = dyn.compute_Fl_and_Fr(u_opt)
    plt.plot_drone(x_opt,u_transformed,TT,simulation_time,x_eq2)

#-----------------------------------------------------------------------------
#TASK 2 trajectory between two positions. (The generated trajectory is smooth)
#-----------------------------------------------------------------------------
if True:
    plt.plot_text("TASK 2 trajectory between two positions.")
    print("TASK 2 trajectory between two positions.")
    ct.set_consider_velocity_control(False)
    simulation_time = 5 #seconds
    visualize_insight = False #if you want to see the plots of the algorithms set it to True

    # initial and final equilibrium points
    x_eq1=np.array([0,0,0,0,0,0,0,0]) #the position is not important is just for the plot
    x_eq2=np.array([1,-1,0,0,0,0,0,0])

    # initialize variables AS THE INITIAL GUESS FOR THE EQUILIBRIUM POINTS SELECT THE CLOSE FORM SOL
    uu_guess_eq1 = np.array([0.66708 , 0.0])
    uu_guess_eq2 = np.array([0.943393 , 0.0])
    TT=math.ceil(simulation_time/dyn.dt)  

    # Compute the input for the initial and final equilibrium points
    u_eq1 = dyn.find_equilibrium_input_xy(uu_guess_eq1,x_eq1[4] , x_eq1[5])
    u_eq2 = dyn.find_equilibrium_input_xy(uu_guess_eq2,x_eq2[4] , x_eq2[5])

    # Define the reference curve
    x_ref, u_ref = plt.get_smooth_reference_curve(x_eq1, u_eq1, x_eq2, u_eq2, TT)

    # Plot the reference curve
    if visualize_insight:
        plt.plot_reference_curve(x_ref, u_ref,TT, simulation_time)

    # Implement the Newton's-like algorithm we will need to rise a flag in order to consider the cost associated
    # with the velocity control
    max_iters = 1000
    armijo_max_iters = 30
    cc=0.7
    beta=0.7
    initial_step_size=0.5
    term_cond = 10**-4
    x_opt,u_opt,_ = slv.newtons_like_algorithm(x_ref, u_ref, max_iters, armijo_max_iters,visualize_insight,term_cond,beta,cc,initial_step_size) 

    #collect this trajectoies for task 3
    x_T3opt = x_opt
    u_T3opt = u_opt
    x_T3ref = x_ref
    u_T3ref = u_ref
    x_T3eq2 = x_eq2

    #show animation
    u_transformed = dyn.compute_Fl_and_Fr(u_opt)
    plt.plot_drone(x_opt,u_transformed,TT,simulation_time,x_ref[:,TT-2])
    

#-----------------------------------------------------------------------------
#TASK 2 trajectory between two velocities. (The generated trajectory is smooth)
#-----------------------------------------------------------------------------
if False:
    plt.plot_text("TASK 2 trajectory between two velocities.")
    print("TASK 2 trajectory between two velocities.")
    ct.set_consider_velocity_control(True)
    simulation_time = 5 #seconds
    visualize_insight = False #if you want to see the plots of the algorithms set it to True

    # initial and final equilibrium points
    x_eq1=np.array([0,-1,0,0,-0.1,0.1  ,0,0]) #the position is not important is just for the plot
    x_eq2=np.array([0,0 ,0,0,0.1 ,0.5,0,0])

    # initialize variables AS THE INITIAL GUESS FOR THE EQUILIBRIUM POINTS SELECT THE CLOSE FORM SOL
    uu_guess_eq1 = np.array([0.66708 , 0.0])
    uu_guess_eq2 = np.array([0.943393 , 0.0])
    TT=math.ceil(simulation_time/dyn.dt)  

    # Compute the input for the initial and final equilibrium points
    u_eq1 = dyn.find_equilibrium_input_vx_vy(uu_guess_eq1,x_eq1[4] , x_eq1[5], x_eq1[3])
    u_eq2 = dyn.find_equilibrium_input_vx_vy(uu_guess_eq2,x_eq2[4] , x_eq2[5], x_eq2[3])

    # Define the reference curve
    x_ref, u_ref = plt.get_smooth_reference_curve(x_eq1, u_eq1, x_eq2, u_eq2, TT)

    # Plot the reference curve
    if visualize_insight:
        plt.plot_reference_curve(x_ref, u_ref,TT, simulation_time)

    # integrate the reference curve velocity to obtain the desired position
    for tt in range(1, TT):
        x_ref[0, tt] = x_ref[0, tt-1] + x_ref[4, tt-1] * dyn.dt
        x_ref[1, tt] = x_ref[1, tt-1] + x_ref[5, tt-1] * dyn.dt

    # Plot the reference curve
    if visualize_insight:
        plt.plot_reference_curve(x_ref, u_ref,TT, simulation_time)

    # Implement the Newton's-like algorithm we will need to rise a flag in order to consider the cost associated
    # with the velocity control
    max_iters = 1000
    armijo_max_iters = 30
    cc=0.7
    beta=0.9
    initial_step_size=0.1
    term_cond = 10**-1
    x_opt,u_opt,_ = slv.newtons_like_algorithm(x_ref, u_ref, max_iters, armijo_max_iters,visualize_insight,term_cond,beta,cc,initial_step_size) 

    #show animation
    u_transformed = dyn.compute_Fl_and_Fr(u_opt)
    plt.plot_drone(x_opt,u_transformed,TT,simulation_time,x_ref[:,TT-2])

#-----------------------------------------------------------------------------
#TASK 3 Tracking performance
#-----------------------------------------------------------------------------
if True:   
    plt.plot_text("TASK 3 Tracking performance of LQR.")
    print("TASK 3 Tracking performance of LQR.")
    ct.set_consider_velocity_control(False)
    TT = x_T3ref.shape[1]

    # Select the displacment from the initial state
    delta_x = np.array([0.2,0.2,-0.1,-0.1,0.1,0.1,0,0])

    # To check the tracking performance of LQR we will use as reference the trajectories generated in the previous 
    # task (task2 trajectory between two pos), linearize the system around x_opt and u_opt and than compute the LQR gain.
    x_tracked, u_tracked, A_opt, B_opt = slv.evaluate_tracking_performance(x_T3ref, u_T3ref, x_T3opt, u_T3opt, delta_x, TT)

    #show animation
    u_transformed = dyn.compute_Fl_and_Fr(u_tracked)
    plt.plot_drone(x_tracked,u_transformed,TT,simulation_time,x_T3eq2)

#-----------------------------------------------------------------------------
#TASK 4 MPC regulation
#-----------------------------------------------------------------------------
if True: 

    plt.plot_text("TASK 4 Model Predictive Control ")

    ns = 8 #8 states 
    ni = 2 #2 inputs 

    xx = np.zeros((ns,1))
    uu = np.zeros((ni,1))  

    xx0 = np.array([0.4,0.4,-0.1,-0.1,0.1,0.1,0,0]) #init value

    T_sim = 500   # simulation horizon
    T_pred = 100 # MPC Prediction horizon

    #creates a 3-dimensional array A_temp
    A_temp = np.zeros((ns,ns,T_pred)) 
    B_temp = np.zeros((ns,ni,T_pred))

    #creates extended matrices and vectors A,B,x_opt and u_opt where the last dimention is T_sim+T_pred
    A_extended = np.zeros((ns, ns, T_sim+T_pred)) 
    B_extended = np.zeros((ns, ni, T_sim+T_pred))
    x_opt_extended = np.zeros((ns, T_sim+T_pred)) 
    u_opt_extended = np.zeros((ni, T_sim+T_pred))

    #assigning the values from A_opt to the first T_sim slices along the extended dimention
    A_extended[:,:,:T_sim] = A_opt 
    B_extended[:,:,:T_sim] = B_opt
    x_opt_extended[:,:T_sim] = x_T3opt 
    u_opt_extended[:,:T_sim] = u_T3opt

    #extend A_extended, B_extended, x_opt_extended, and u_opt_extended arrays 
    #by copying the last slice or column of A_opt, B_opt, x_T3opt, and u_T3opt 
    #respectively into the next T_pred slices or columns, in order to perform 
    #the last iterations of the MPC 
    for tt in range(T_pred): 
        A_extended[:,:,T_sim+tt] = A_opt[:,:,-1]
        B_extended[:,:,T_sim+tt] = B_opt[:,:,-1]
        x_opt_extended[:,T_sim+tt] = x_T3opt[:,-1]
        u_opt_extended[:,T_sim+tt] = u_T3opt[:,-1]


    # Definition of state cost 
    QQ = [[9, 0, 0, 0, 0, 0, 0, 0],
          [0, 9, 0, 0, 0, 0, 0, 0],
          [0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0],
          [0, 0, 0, 0, 0, 1, 0, 0],
          [0, 0, 0, 0, 0, 0, 5, 0],
          [0, 0, 0, 0, 0, 0, 0, 5]]
    
    RR = [[3, 0],
          [0, 3]]
    
    QQf = QQ
    
    #############################
    # Model Predictive Control
    #############################
    
    x_real_mpc = np.zeros((ns,T_sim))
    u_real_mpc = np.zeros((ni,T_sim))

    x_mpc = np.zeros((ns, T_pred, T_sim))

    x_real_mpc[:,0] = xx0.squeeze()

    #at each iteration we set the initial state as the current state of the real system and solve the mpc by calling the linear mpc function of the solver.py and then we apply first input to the dynamics
    for tt in range(T_sim-1): 
        # System evolution - real with MPC
        x_t_mpc = x_real_mpc[:,tt] # get initial condition

        # Solve MPC problem - apply first input
        if tt%5 == 0: # print every 5 time instants
            print('MPC:\t t = {} of 500'.format(tt))

        #get the first T_pred slices along the third dimension of matrices starting from tt
        A_temp = A_extended[:,:,tt:tt+T_pred] 
        B_temp = B_extended[:,:,tt:tt+T_pred]
        #get the first T_pred columns of vectors starting from tt
        x_opt_temp = x_opt_extended[:,tt:tt+T_pred] 
        u_opt_temp = u_opt_extended[:,tt:tt+T_pred]

        u_real_mpc[:,tt], x_mpc[:,:,tt] = slv.linear_mpc(A_temp, B_temp, QQ, RR, QQf, x_t_mpc, x_opt_temp, u_opt_temp, T_pred)[:2]
        
        x_real_mpc[:,tt+1] = dyn.dynamics(x_real_mpc[:,tt], u_real_mpc[:,tt])[0]

    plt.plot_mpc(T_sim, ns, ni, x_real_mpc, x_T3opt, u_real_mpc, u_T3opt)

    plt.plot_tracking_error(xx0, x_real_mpc, u_real_mpc, x_T3opt,u_T3opt, ns,ni, T_sim, T_pred)