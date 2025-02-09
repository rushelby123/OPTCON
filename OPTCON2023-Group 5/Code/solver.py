import numpy as np
import cost as ct   
import dynamics as dyn  
import plots as plt
import cvxpy as cp
import matplotlib.pyplot as pyplt

def ltv_LQR(AAin, BBin, QQin, RRin, SSin, QQfin, TT, x0, qqin = None, rrin = None, qqfin = None):

  """
	LQR for LTV system with (time-varying) affine cost
	
  Args
    - AAin (nn x nn (x TT)) matrix
    - BBin (nn x mm (x TT)) matrix
    - QQin (nn x nn (x TT)), RR (mm x mm (x TT)), SS (mm x nn (x TT)) stage cost
    - QQfin (nn x nn) terminal cost
    - qq (nn x (x TT)) affine terms
    - rr (mm x (x TT)) affine terms
    - qqf (nn x (x TT)) affine terms - final cost
    - TT time horizon
  Return
    - KK (mm x nn x TT) optimal gain sequence
    - PP (nn x nn x TT) riccati matrix
  """
	
  try:
    # check if matrix is (.. x .. x TT) - 3 dimensional array 
    ns, lA = AAin.shape[1:]
  except:
    # if not 3 dimensional array, make it (.. x .. x 1)
    AAin = AAin[:,:,None]
    ns, lA = AAin.shape[1:]

  try:  
    ni, lB = BBin.shape[1:]
  except:
    BBin = BBin[:,:,None]
    ni, lB = BBin.shape[1:]

  try:
      nQ, lQ = QQin.shape[1:]
  except:
      QQin = QQin[:,:,None]
      nQ, lQ = QQin.shape[1:]

  try:
      nR, lR = RRin.shape[1:]
  except:
      RRin = RRin[:,:,None]
      nR, lR = RRin.shape[1:]

  try:
      nSi, nSs, lS = SSin.shape
  except:
      SSin = SSin[:,:,None]
      nSi, nSs, lS = SSin.shape

  # Check dimensions consistency -- safety
  if nQ != ns:
    print("Matrix Q does not match number of states")
    exit()
  if nR != ni:
    print("Matrix R does not match number of inputs")
    exit()
  if nSs != ns:
    print("Matrix S does not match number of states")
    exit()
  if nSi != ni:
    print("Matrix S does not match number of inputs")
    exit()


  if lA < TT:
    AAin = AAin.repeat(TT, axis=2)
  if lB < TT:
    BBin = BBin.repeat(TT, axis=2)
  if lQ < TT:
    QQin = QQin.repeat(TT, axis=2)
  if lR < TT:
    RRin = RRin.repeat(TT, axis=2)
  if lS < TT:
    SSin = SSin.repeat(TT, axis=2)

  # Check for affine terms

  augmented = False

  KK = np.zeros((ni, ns, TT))
  sigma = np.zeros((ni, TT))
  PP = np.zeros((ns, ns, TT))
  pp = np.zeros((ns, TT))
  QQ = QQin
  RR = RRin
  SS = SSin
  QQf = QQfin
  qq = qqin
  rr = rrin
  qqf = qqfin
  AA = AAin
  BB = BBin
  xx = np.zeros((ns, TT))
  uu = np.zeros((ni, TT))

  xx[:,0] = x0
  
  PP[:,:,-1] = QQf
  pp[:,-1] = qqf
  
  # Solve Riccati equation
  for tt in reversed(range(TT-1)):
    QQt = QQ[:,:,tt]
    qqt = qq[:,tt][:,None]
    RRt = RR[:,:,tt]
    rrt = rr[:,tt][:,None]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    SSt = SS[:,:,tt]
    PPtp = PP[:,:,tt+1]
    pptp = pp[:, tt+1][:,None]

    MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
    mmt = rrt + BBt.T @ pptp
    
    PPt = AAt.T @ PPtp @ AAt - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ (BBt.T@PPtp@AAt + SSt) + QQt
    ppt = AAt.T @ pptp - (BBt.T@PPtp@AAt + SSt).T @ MMt_inv @ mmt + qqt

    PP[:,:,tt] = PPt
    pp[:,tt] = ppt.squeeze()


  # Evaluate KK
  
  for tt in range(TT-1):
    QQt = QQ[:,:,tt]
    qqt = qq[:,tt][:,None]
    RRt = RR[:,:,tt]
    rrt = rr[:,tt][:,None]
    AAt = AA[:,:,tt]
    BBt = BB[:,:,tt]
    SSt = SS[:,:,tt]

    PPtp = PP[:,:,tt+1]
    pptp = pp[:,tt+1][:,None]

    # Check positive definiteness

    MMt_inv = np.linalg.inv(RRt + BBt.T @ PPtp @ BBt)
    mmt = rrt + BBt.T @ pptp

    # for other purposes we could add a regularization step here...????????------------------------

    KK[:,:,tt] = -MMt_inv@(BBt.T@PPtp@AAt + SSt)
    sigma_t = -MMt_inv@mmt

    sigma[:,tt] = sigma_t.squeeze()


  

  for tt in range(TT - 1):
    # Trajectory

    uu[:, tt] = KK[:,:,tt]@xx[:, tt] + sigma[:, tt]
    xx_p = AA[:,:,tt]@xx[:,tt] + BB[:,:,tt]@uu[:, tt]

    xx[:,tt+1] = xx_p

    xxout = xx
    uuout = uu

  return KK, sigma, PP, xxout, uuout

def newtons_like_algorithm(x_ref, u_ref, max_iters,max_iters_armijo,visu_armijo,term_cond,beta,cc,initial_step_size):
  """
      newton_method_for_optimal_control_problem, for a quadrator system

      Args
      - xx_ref desired state trajectory
      - uu_ref desired input trajectory
      - ns number of states
      - ni number of inputs
      - max_iters maximum number of iterations
      - visu_armijo boolean to visualize the armijo rule

      Return 
      -  uu_opt optimal input trajectory

  """
  ni = dyn.ni
  ns = dyn.ns

# Initialize the variables
  TT=x_ref.shape[1]
  JJ=np.zeros(max_iters)
  dJ=np.zeros((ni,TT,max_iters))
  xx = np.zeros((ns,TT,max_iters))
  uu = np.zeros((ni,TT,max_iters))
  QQT = np.zeros((ns,ns))  
  QQ = np.zeros((ns,ns,TT))   
  RR = np.zeros((ni,ni,TT))   
  SS = np.zeros((ni,ns,TT))   #hessian of l wrt u,x (in basso a sinistra)
  delta_x = np.zeros((ns,TT))
  delta_u = np.zeros((ni,TT))
  lmbd = np.zeros((ns, TT, max_iters))
  descent_arm = np.zeros(max_iters)
  descent = np.zeros(max_iters)
  At = np.zeros((ns,ns,TT))
  Bt = np.zeros((ns,ni,TT))
  at = np.zeros((ns,TT))
  bt = np.zeros((ni,TT))
  K_opt=np.zeros((ni, ns, TT))
  sigma_opt=np.zeros((ni, TT))

  # Initialize the Trajectories
  xx[:,:,0] = (x_ref[:,0])[:, None]
  uu[:,:,0] = u_ref[:,0][:, None]
  for i in range (max_iters):
    xx[:,0,i] = x_ref[:,0]

  #init the input sequence
  for tt in range(TT-1):
    uu[:,tt,0] = u_ref[:,0]

  # Newton's method
  for i in range (0,max_iters-1):     

    #compute the costate equation and the linearization of the system about the current trajectory
    lmbd_temp = ct.termcost(xx[:,TT-1,i], x_ref[:,TT-1])[1]
    lmbd[:,TT-1,i] = lmbd_temp.squeeze()
    for tt in reversed(range(TT-1)):
      at[:,tt], bt[:,tt], QQ[:,:,tt], _, RR[:,:,tt], SS[:,:,tt] = ct.stagecost(xx[:,tt, i], uu[:,tt,i], x_ref[:,tt], u_ref[:,tt])[1:]
      fx, fu = dyn.dynamics(xx[:,tt, i], uu[:,tt,i])[1:]
      At[:,:,tt] = fx.T
      Bt[:,:,tt] = fu.T
      lmbd_temp = At[:,:,tt].T@lmbd[:,tt+1,i] + at[:,tt]     
      dJ_temp   = -Bt[:,:,tt].T@lmbd[:,tt+1,i] - bt[:,tt]      
      lmbd[:,tt,i] = lmbd_temp.squeeze()
      dJ[:,tt,i] = dJ_temp.squeeze()
      #delta_u[:,tt] = -dJ_temp #steppest descent direction
    QQT = ct.termcost(xx[:,-1,i], x_ref[:,-1])[2]
    
    #compute the cost 
    JJ[i] = 0
    for tt in range(TT-1):
        JJ[i] += ct.stagecost(xx[:,tt,i], uu[:,tt,i], x_ref[:,tt], u_ref[:,tt])[0]
    JJ[i] += ct.termcost(xx[:,-1,i], x_ref[:,-1])[0]
    print(f'JJ[{i}] = {JJ[i]:.5f}')
    
    #compute the delta_x and delta_u using affine time variant LQR used in lab session 4
    K_opt,sigma_opt,_,_,delta_u = ltv_LQR(At, Bt, QQ, RR, SS, QQT, TT,0, at, bt, at[:,-1]) #delta x0 = 0 always
      
    #compute the descent direction for armijo
    for tt in range(TT-1):
      descent_arm[i] += dJ[:,tt,i].T@delta_u[:,tt]    
      descent[i] += delta_u[:,tt].T@delta_u[:,tt]      
      
    #armijo stepsize selection
    stepsize=get_step_size(xx, uu, x_ref, u_ref, descent_arm[i], K_opt, sigma_opt, max_iters_armijo, beta, cc, visu_armijo,i,delta_u,initial_step_size,JJ[i])

    #update input sequence using the "descent" trajectory and a closed loop LQR controller
    for tt in range(TT-1):
      delta_x[:,tt]=xx[:,tt,i+1]-xx[:,tt,i]       #delta_x is the displacement/error from the "descent" trajectory
      uu[:,tt,i+1]=uu[:,tt,i]+sigma_opt[:,tt]*stepsize+K_opt[:,:,tt]@delta_x[:,tt] #compute the new input sequence
      xx[:,tt+1,i+1] = dyn.dynamics(xx[:,tt,i+1], uu[:,tt,i+1])[0]                 #compute the new state sequence

    #plot the results
    if visu_armijo==True:
      plt.plot_curves_with_reference(x_ref,u_ref,xx[:,:,i], uu[:,:,i],TT, 10)
    print(f'Total cost J[{i}] = {JJ[i]:.5f} Descent[{i}] = {descent_arm[i]:.5f} Armijo stepsize = {stepsize}')

    #terminal condition 
    if abs(descent_arm[i]) <= term_cond :
      print('terminal condition satisfied ')
      plt.plot_descent_and_cost(descent_arm, JJ,i)
      plt.plot_curves_with_reference(x_ref,u_ref,xx[:,:,i], uu[:,:,i],TT, 10)
      return xx[:,:,i],uu[:,:,i], JJ[i]
      break
  
  #if the maximum number of iterations is reached return the last computed trajectory
  plt.plot_descent_and_cost(descent, JJ,max_iters)
  plt.plot_curves_with_reference(x_ref,u_ref,xx[:,:,max_iters], uu[:,:,max_iters],TT, 10)
  return xx[:,:,-1],uu[:,:,-1], JJ[-1]

def get_step_size(xx, uu, x_ref, u_ref, descent_arm, KK, sigma, max_iters_armijo, beta,cc, visu_armijo,i,delta_u,initial_step_size,JJ):
  #define variables
  TT=x_ref.shape[1]
  steps = np.linspace(0,1,int(1e1))
  costs = np.zeros(len(steps))
  x0=xx[:,0,i]

  #compute cost with stepsize 0
  xx_temp = np.zeros((dyn.ns,TT))
  uu_temp = np.zeros((dyn.ni,TT))
  delta_x = np.zeros((dyn.ns,TT))
  xx_temp[:,0] = x0
  for tt in range(TT-1):
    #uu_temp[:,tt] = uu[:,tt,i] + step*delta_u[:,tt]
    #xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]
    delta_x[:,tt]=xx_temp[:,tt]-xx[:,tt,i]       #delta_x is the displacement/error from the "descent" trajectory
    uu_temp[:,tt]=uu[:,tt,i]+KK[:,:,tt]@delta_x[:,tt] #compute the new input sequence
    xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]                 #compute the new state sequence
  JJ_temp = 0
  for tt in range(TT-1):
    temp_cost = ct.stagecost(xx_temp[:,tt], uu_temp[:,tt], x_ref[:,tt], u_ref[:,tt])[0]
    JJ_temp += temp_cost
  temp_cost = ct.termcost(xx_temp[:,-1], x_ref[:,-1])[0]
  JJ_temp += temp_cost
  JJ0 = JJ_temp

  #compute the armijo's costs
  stepsizes = []  # list of stepsizes
  costs_armijo = []
  stepsize = initial_step_size
  for ii in range(max_iters_armijo):
    # temp solution update
    xx_temp = np.zeros((dyn.ns,TT))
    uu_temp = np.zeros((dyn.ni,TT))
    delta_x = np.zeros((dyn.ns,TT))
    xx_temp[:,0] = x0
    for tt in range(TT-1):
      delta_x[:,tt]=xx_temp[:,tt]-xx[:,tt,i]       #delta_x is the displacement/error from the "descent" trajectory
      uu_temp[:,tt]=uu[:,tt,i]+sigma[:,tt]*stepsize+KK[:,:,tt]@delta_x[:,tt] #compute the new input sequence
      xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]                 #compute the new state sequence

    # temp cost calculation
    JJ_temp = 0
    for tt in range(TT-1):
      temp_cost = ct.stagecost(xx_temp[:,tt], uu_temp[:,tt], x_ref[:,tt], u_ref[:,tt])[0]
      JJ_temp += temp_cost
    temp_cost = ct.termcost(xx_temp[:,-1], x_ref[:,-1])[0]
    JJ_temp += temp_cost
    stepsizes.append(stepsize)      # save the stepsize
    costs_armijo.append(JJ_temp)    # save the cost associated to the stepsize
    if JJ_temp > JJ0 - cc*stepsize*descent_arm:
        # update the stepsize
        stepsize = beta*stepsize
    else:
        print('Armijo stepsize = {}'.format(stepsize))
        break


  if visu_armijo and not i==0:
    #compute the cost for each stepsize
    for ii in range(len(steps)):
      step = steps[ii]
      xx_temp = np.zeros((dyn.ns,TT))
      uu_temp = np.zeros((dyn.ni,TT))
      delta_x = np.zeros((dyn.ns,TT))
      xx_temp[:,0] = x0
      for tt in range(TT-1):
        #uu_temp[:,tt] = uu[:,tt,i] + step*delta_u[:,tt]
        #xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]
        delta_x[:,tt]=xx_temp[:,tt]-xx[:,tt,i]       #delta_x is the displacement/error from the "descent" trajectory
        uu_temp[:,tt]=uu[:,tt,i]+sigma[:,tt]*step+KK[:,:,tt]@delta_x[:,tt] #compute the new input sequence
        xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]                 #compute the new state sequence

      JJ_temp = 0
      for tt in range(TT-1):
        temp_cost = ct.stagecost(xx_temp[:,tt], uu_temp[:,tt], x_ref[:,tt], u_ref[:,tt])[0]
        JJ_temp += temp_cost
      temp_cost = ct.termcost(xx_temp[:,-1], x_ref[:,-1])[0]
      JJ_temp += temp_cost
      costs[ii] = JJ_temp

    # Create a plot
    pyplt.figure()
    pyplt.title('Armijo rule visualization')
    pyplt.plot(stepsizes, costs_armijo, 'x', label='Armijo_Costs')
    pyplt.plot(steps, costs,'gold', label='Cost')
    pyplt.plot(steps, JJ - steps * descent_arm, 'g-', label='Tangent line')
    pyplt.plot(steps, JJ - cc * steps* descent_arm, 'r--', label='Armijo line')

    # Add labels
    pyplt.xlabel('stepsize')
    pyplt.ylabel('Cost')

    # Add a legend
    pyplt.legend()

    # Show the plot
    pyplt.show()
  
  return stepsize

def evaluate_tracking_performance(x_ref, u_ref, x_opt, u_opt, delta_init, TT):
  #initialize variables
  ns=dyn.ns
  ni=dyn.ni 
  QQT = np.zeros((ns,ns)) 
  At = np.zeros((ns,ns,TT))  
  Bt = np.zeros((ns,ni,TT))  
  QQ = np.zeros((ns,ns,TT))   
  RR = np.zeros((ni,ni,TT))   
  SS = np.zeros((ni,ns,TT))   #hessian of l wrt u,x (in basso a sinistra)
  at = np.zeros((ns,TT))  
  bt = np.zeros((ni,TT))  
  K_opt=np.zeros((ni, ns, TT))
  x_tracked = np.zeros((ns,TT))
  u_tracked = np.zeros((ni,TT))
  state_error = np.zeros((ns,TT)) 
  x_tracked[:,0] = x_opt[:,0]+delta_init

  #compute the lineazation of the system about the optimal trajectory computed in task 3
  #1-compute at=qt bt=rt Qt Rt St forward in time 
  for tt in range(TT-1): #Q AND R ARE DEGREE OF FREEDOM 
    at[:,tt], bt[:,tt],QQ[:,:,tt],_,RR[:,:,tt],SS[:,:,tt] = ct.stagecost(x_opt[:,tt], u_opt[:,tt], x_ref[:,tt], u_ref[:,tt])[1:]
    fx, fu = dyn.dynamics(x_opt[:,tt], u_opt[:,tt])[1:]
    At[:,:,tt] = fx.T
    Bt[:,:,tt] = fu.T
  at[:,-1],QQT = ct.termcost(x_ref[:,-1], x_ref[:,-1])[1:]  

  #compute the kalman gain for each time step of the linearized trajectory
  delta_x = delta_init 
  K_opt,_,_,_,_=ltv_LQR(At, Bt, QQ, RR, SS, QQT, TT,delta_x, at, bt, at[:,-1]) #if i set q, r and qf to none then error arise

  #compute the new input sequence with the closed loop controller
  for tt in range(TT-1):
    #evaluate the error between the optimal trajectory and the tracked trajectory
    state_error[:,tt] = x_tracked[:,tt] - x_opt[:,tt]
    
    #compute the new input sequence
    u_tracked[:,tt] = u_opt[:,tt]+K_opt[:,:,tt]@state_error[:,tt]
    
    #compute the trajectory forward in time
    x_tracked[:,tt+1] = dyn.dynamics(x_tracked[:,tt], u_tracked[:,tt])[0]
  
  #plot the trajectory 
  plt.plot_tracking_performance(x_ref,u_ref,x_opt, u_opt,x_tracked, u_tracked,TT, 10)
  
  #return the tracked trajectory
  return x_tracked, u_tracked, At, Bt

def linear_mpc(AA_opt, BB_opt, QQ, RR, QQf, xxt, xx_opt, uu_opt, T_pred):

  xxt = xxt.squeeze()

  ns, ni = BB_opt.shape[:2]

  xx_mpc = cp.Variable((ns, T_pred))
  uu_mpc = cp.Variable((ni, T_pred))

  cost = 0
  constr = []

  for tt in range(T_pred-1):
    cost += cp.quad_form(xx_mpc[:,tt] - xx_opt[:,tt], QQ) + cp.quad_form(uu_mpc[:,tt] - uu_opt[:,tt], RR) #to track optimal trajectory we try to minimize the difference between the optimal trajectory and the predicted trajectory  
    constr += [xx_mpc[:,tt+1] - xx_opt[:,tt+1] == (AA_opt[:,:,tt]@(xx_mpc[:,tt] - xx_opt[:,tt]) + BB_opt[:,:,tt]@(uu_mpc[:,tt]-uu_opt[:,tt]))] # dynamics constraint considering the difference of optimal and predicted trajectory
            
  # sums problem objectives and concatenates constraints.
  cost += cp.quad_form(xx_mpc[:,T_pred-1] - xx_opt[:, T_pred-1], QQf) #invece di Qf , P della riccati eq
  
  constr += [xx_mpc[:,0] - xx_opt[:,0] == xxt - xx_opt[:,0]] #list of constraints CHECK IF CORRECT xxt - xx_opt
  
  problem = cp.Problem(cp.Minimize(cost), constr)
  problem.solve()
  print("cost", problem.value)


  if problem.status == "infeasible":
  # Otherwise, problem.value is inf or -inf, respectively.
    print("Infeasible problem! CHECK YOUR CONSTRAINTS!!!")

  return uu_mpc[:,0].value, xx_mpc.value, uu_mpc.value


   