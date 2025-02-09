import numpy as np
import dynamics as dyn

ns = dyn.ns
ni = dyn.ni

#important this variable will be used in the main to select wich cost function to use!!
consider_velocity_control = None

def set_consider_velocity_control(value):
    global consider_velocity_control
    consider_velocity_control = value

#costs for position control
QQt = [ [10, 0, 0, 0, 0, 0, 0, 0],
        [0, 10, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]
      ]
RRt = [ [3, 0],
        [0, 3]
      ]
QQT = QQt

#costs for velocity control
QQt_vel = [ [0.0001, 0, 0, 0, 0, 0, 0, 0],#x
            [0, 0.0001, 0, 0, 0, 0, 0, 0],#y
            [0, 0, 0.1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0.1, 0, 0, 0, 0],
            [0, 0, 0, 0, 10, 0, 0, 0],#vx
            [0, 0, 0, 0, 0, 10, 0, 0],#vy
            [0, 0, 0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 0, 0, 2]
        ]
RRt_vel = [ [3, 0],
            [0, 3]
      ]
QQT_vel = QQt_vel

def stagecost(xx,uu, xx_ref, uu_ref):
  """
    Stage-cost 

    Quadratic cost function 
    l(x,u) = 1/2 (x - x_ref)^T Q (x - x_ref) + 1/2 (u - u_ref)^T R (u - u_ref)

    Args
      - xx \in \R^n state at time t
      - xx_ref \in \R^n state reference at time t

      - uu \in \R^m input at time t
      - uu_ref \in \R^m input reference at time t
      
      - QQt \in \R^{n x n} state cost matrix
      - RRt \in \R^{m x m} input cost matrix

    Return 
      - cost at xx,uu
      - gradient of l wrt x, at xx,uu
      - gradient of l wrt u, at xx,uu
      - hessian of l wrt x,x, at xx,uu
      - hessian of l wrt x,u, at xx,uu
      - hessian of l wrt u,u, at xx,uu
      - hessian of l wrt u,x, at xx,uu
  
  """
  if not consider_velocity_control:
    ll = 0.5*(xx - xx_ref).T@QQt@(xx - xx_ref) + 0.5*(uu - uu_ref).T@RRt@(uu - uu_ref)

    lx = QQt@(xx - xx_ref)
    lu = RRt@(uu - uu_ref)

    lxx = QQt
    lxu = np.zeros((ns,ni)) 

    luu = RRt
    lux = np.zeros((ni,ns))
  
  else:
    ll = 0.5*(xx - xx_ref).T@QQt_vel@(xx - xx_ref) + 0.5*(uu - uu_ref).T@RRt_vel@(uu - uu_ref)

    lx = QQt_vel@(xx - xx_ref)
    lu = RRt_vel@(uu - uu_ref)

    lxx = QQt_vel
    lxu = np.zeros((ns,ni)) 

    luu = RRt_vel
    lux = np.zeros((ni,ns))
  
  return ll, lx, lu, lxx, lxu, luu, lux

def termcost(xx,xx_ref):
  """
    Terminal-cost

    Quadratic cost function l_T(x) = 1/2 (x - x_ref)^T Q_T (x - x_ref)

    Args
      - xx \in \R^2 state at time t
      - xx_ref \in \R^2 state reference at time t

      - QQT \in \R^{n x n} state cost matrix

    Return 
      - cost at xx,uu
      - gradient of l wrt x, at xx,uu
      - gradient of l wrt u, at xx,uu
  
  """
  if not consider_velocity_control:
    llT = 0.5*(xx - xx_ref).T@QQT@(xx - xx_ref)

    lTx = QQT@(xx - xx_ref)

    lTxx = QQT
  else:

    llT = 0.5*(xx - xx_ref).T@QQT_vel@(xx - xx_ref)

    lTx = QQT_vel@(xx - xx_ref)

    lTxx = QQT_vel

  return llT, lTx, lTxx
