# Discrete-time nonlinear dynamics for the planar quadrotor
import numpy as np
from scipy.optimize import fsolve
import cvxpy as cvx

ns = 8 #8 states 
ni = 2 #2 inputs 
dt = 1e-2 # discretization stepsize - Forward Euler

# Dynamics parameters
mm_drone = 0.028
mm_load = 0.04
mm_tot = mm_load + mm_drone
JJ = 0.001
ll_drone = 0.05 #is this half the drone? distance from center of mass to propeller 
ll_load = 0.2
gg = 9.81

def dynamics(xx,uu):
  """
    Nonlinear dynamics of a quadrotor

    Args
      - xx \in \R^8 state at time t
      - uu \in \R^2 input at time t

    Return 
      - next state xx_{t+1}
      - gradient of f wrt x, at xx,uu
      - gradient of f wrt u, at xx,uu
  
  """
  xx = xx[:,None]
  uu = uu[:,None]

  xxp = np.zeros((ns,1))

  #dynamics
  xxp[0] = xx[0,0] + dt * xx[4,0] #xp' = vx
  xxp[1] = xx[1,0] + dt * xx[5,0] #yp' = vy
  xxp[2] = xx[2,0] + dt * xx[6,0] #alpha' = Walpha
  xxp[3] = xx[3,0] + dt * xx[7,0] #theta' = Wtheta
  xxp[4] = xx[4,0] + dt* (mm_load * ll_load * (xx[6,0]**2) * np.sin(xx[2,0]) - uu[0,0] * (np.sin(xx[3,0]) - (mm_load/mm_drone) * np.sin(xx[2,0] - xx[3,0]) * np.cos(xx[2,0]))) /mm_tot # vx'
  xxp[5] = xx[5,0] + dt* ((-mm_load * ll_load * (xx[6,0]**2) * np.cos(xx[2,0]) + uu[0,0] * (np.cos(xx[3,0]) + (mm_load/mm_drone) * np.sin(xx[2,0] - xx[3,0]) * np.sin(xx[2,0]))- (mm_tot)*gg)/mm_tot) # vx'
  xxp[6] = xx[6,0] + dt* (-uu[0,0] * np.sin(xx[2,0] - xx[3,0])) / (mm_drone * ll_load) #Walpha'
  xxp[7] = xx[7,0] + dt* ll_drone * uu[1,0] / JJ #Wtheta'

  # Gradient
  fx = np.zeros((ns, ns))
  fu = np.zeros((ni, ns))

  #df1 OK OK
  fx[0,0] = 1
  fx[1,0] = 0
  fx[2,0] = 0
  fx[3,0] = 0
  fx[4,0] = dt #delta
  fx[5,0] = 0
  fx[6,0] = 0
  fx[7,0] = 0

  fu[0,0] = 0
  fu[1,0] = 0

  #df2 OK OK
  fx[0,1] = 0
  fx[1,1] = 1
  fx[2,1] = 0
  fx[3,1] = 0
  fx[4,1] = 0
  fx[5,1] = dt #delta
  fx[6,1] = 0
  fx[7,1] = 0

  fu[0,1] = 0 
  fu[1,1] = 0

  #df3
  fx[0,2] = 0
  fx[1,2] = 0
  fx[2,2] = 1
  fx[3,2] = 0
  fx[4,2] = 0
  fx[5,2] = 0
  fx[6,2] = dt #delta
  fx[7,2] = 0

  fu[0,2] = 0
  fu[1,2] = 0

  #df4
  fx[0,3] = 0
  fx[1,3] = 0
  fx[2,3] = 0
  fx[3,3] = 1
  fx[4,3] = 0
  fx[5,3] = 0
  fx[6,3] = 0
  fx[7,3] = dt #delta

  fu[0,3] = 0
  fu[1,3] = 0

  #df5 OK OK
  fx[0,4] = 0 
  fx[1,4] = 0
  fx[2,4] = (dt /mm_tot) * (mm_load * ll_load * xx[6,0]**2 * np.cos(xx[2,0]) + uu[0,0] * mm_load /mm_drone * (np.cos(xx[2,0] - xx[3,0]) * np.cos(xx[2,0]) - np.sin(xx[2,0]-xx[3,0])*np.sin(xx[2,0]))) 
  fx[3,4] =-(dt * uu[0,0] /mm_tot)* (np.cos(xx[3,0]) + (mm_load/ mm_drone) * np.cos(xx[2,0] - xx[3,0]) * np.cos(xx[2,0])  ) 
  fx[4,4] = 1
  fx[5,4] = 0
  fx[6,4] = 2* dt * mm_load * ll_load * xx[6,0] * np.sin(xx[2,0]) / mm_tot 
  fx[7,4] = 0

  fu[0,4] =  -dt/mm_tot*(np.sin(xx[3,0])-mm_load/mm_drone*np.sin(xx[2,0]-xx[3,0])*np.cos(xx[2,0]))
  fu[1,4] = 0

  #df6 OK OK
  fx[0,5] = 0
  fx[1,5] = 0
  fx[2,5] = dt/mm_tot*(mm_load*ll_load*(xx[6,0]**2)*np.sin(xx[2,0])+uu[0,0]*(mm_load/mm_drone)*np.cos(xx[2,0]-xx[3,0])*np.sin(xx[2,0])+np.sin(xx[2,0]-xx[3,0])*np.cos(xx[2,0]))
  fx[3,5] = dt/mm_tot*uu[0,0]*(-np.sin(xx[3,0])-(mm_load/mm_drone)*np.cos(xx[2,0]-xx[3,0])*np.sin(xx[2,0]))
  fx[4,5] = 0
  fx[5,5] = 1
  fx[6,5] = - (dt/mm_tot) * 2 * (mm_load * ll_load * np.cos(xx[2,0]) * xx[6,0])
  fx[7,5] = 0

  fu[0,5] = dt/mm_tot*(np.cos(xx[3,0])+(mm_load/mm_drone)*np.sin(xx[2,0]-xx[3,0])*np.sin(xx[2,0]))
  fu[1,5] = 0

  #df7 OK OK
  fx[0,6] = 0 
  fx[1,6] = 0 
  fx[2,6] = -dt * uu[0,0] * np.cos(xx[2,0] - xx[3,0]) / (mm_drone * ll_load)
  fx[3,6] =  dt * uu[0,0] * np.cos(xx[2,0] - xx[3,0]) / (mm_drone * ll_load)
  fx[4,6] = 0 
  fx[5,6] = 0 
  fx[6,6] = 1
  fx[7,6] = 0

  fu[0,6] = -(dt/(mm_drone*ll_load))*np.sin(xx[2,0]-xx[3,0])
  fu[1,6] = 0

  #df8 OK OK
  fx[0,7] = 0
  fx[1,7] = 0
  fx[2,7] = 0
  fx[3,7] = 0
  fx[4,7] = 0
  fx[5,7] = 0
  fx[6,7] = 0
  fx[7,7] = 1

  fu[0,7] = 0
  fu[1,7] = dt*ll_drone/JJ

  xxp = xxp.squeeze()
  return xxp, fx, fu #return x(t+1), A and B transpose

def find_equilibrium_input_xy(uu_guess,x_des,y_des):
  """
    Finds the input that makes the system in equilibrium in a point defind by coordinates
    x_des and y_des

    Args
      - uu_guess \in \R^2 initial guess for the input
      - x_des \in \R^8 desired equlibrium state

    Return 
      - input uu such that xx are equilibrium points of the dynamics
    
    Note that the desired state x_des is not necessarily an equilibrium point it'
  """ 
  #check if the desired state is an equilibrium point
  x_des = np.array([x_des,y_des,0,0,0,0,0,0])
  
  def functions(uu):
    xp=dynamics(x_des,uu)[0]  #get the dynamic equations from the function "dynamics" above
    equations=np.array([xp[5],xp[7]]) #set the equations to be solved for the equilibrium point
    return equations.squeeze() 
    
  uu_eq = fsolve(functions,uu_guess) #find the input for the desired configuration
  return uu_eq

def find_equilibrium_input_vx_vy(uu_guess,vx_des,vy_des,theta_des):
  """
    Finds the input that makes the system in equilibrium while moving at a constant velocity defind by coordinates  
    x_des[4] and x_des[5]

    Args
      - uu_guess \in \R^2 initial guess for the input
      - x_des \in \R^8 desired equlibrium state

    Return 
      - input uu such that xx are equilibrium points of the dynamics
    
    Note that the desired state x_des is not necessarily an equilibrium point it'
  """ 
  x_des=np.array([0,0,theta_des,theta_des,vx_des,vy_des,0,0]) 

  def functions(uu):
    xp=dynamics(x_des,uu)[0]  #get the dynamic equations from the function "dynamics" above
    equations=np.array([xp[5]-vy_des,xp[7]]) #set the equations to be solved for the equilibrium point
    return equations.squeeze() 
    
  uu_eq = fsolve(functions,uu_guess) #find the input for the desired configuration
  return uu_eq

def compute_Fl_and_Fr(uu):
  """
    Computes the thrust forces of the left and right propellers

    Args
      - uu \in [ni,TT] (center thrust force and torque)

    Return 
      - uu \in [ni,TT] input at time t (left and right thrust forces)
  """ 
  #initialize the input vector
  uu_ref = np.zeros((ni,uu.shape[1]))
  #for each time step compute the thrust forces of the left and right propellers
  for tt in range(uu.shape[1]): 
    Fl = (uu[0,tt]+uu[1,tt])/2
    Fr = (uu[0,tt]-uu[1,tt])/2
    uu_ref[0,tt] = Fl 
    uu_ref[1,tt] = Fr
  return uu_ref