# %% grab dependencies 
import serial
import math
import numpy as np
import time
import theano.tensor as T
import matplotlib.pyplot as plt

from ilqr import iLQR 
from ilqr.cost import QRCost
from ilqr.dynamics import AutoDiffDynamics, tensor_constrain, constrain

# %% define functions

def motor_test():
    user_input = input("\n <Motor 1 Speed, Motor2 Speed>: ")
    if user_input == "quit" or user_input == "q":
        ser.close()
    elif user_input == "s" or user_input == "stop":
        ser.write(str.encode("<0,0>"))
        motor_test()
    else:
        ser.write(str.encode(user_input))
        motor_test()
def run_trajectory(X,U,dt):
    for row in X:
        lspeed = int(row[2]*micro_step*pulley_ratio/(2*wheel_radius*np.pi))
        tspeed = int(row[3]*micro_step*gear_ratio/(2*np.pi))
        command = '<' + repr(lspeed)+',' + repr(tspeed)+'>'
        print(command)
        ser.write(str.encode(command))
        time.sleep(dt)
        
def length_traj(T,l0,lf,theta0, lfdot):
    tf = T[-1]
    n = len(T)
    ns = math.ceil(n/2)
    ts = T[ns-1]
    T1 = T[0:ns]
    T2 = T[ns:]
    m = m_end + mboom
    F1 = 2*m*(2*l0-2*lf+lfdot*tf/2)/tf**2
    F2 = 2*m*(2*l0-2*lf+3*lfdot*tf/2)/tf**2
    theta = theta0*np.ones((len(T),1))
    l1 = -(F1*T1**2)/(m*2) + l0
    l2 = -(F1*ts**2)/(m*2) + l0 - (F1/m)*ts*(T2-ts) + (F2/(2*m))*(T2-ts)**2
    l = np.concatenate((l1,l2),axis =0)
    l = l.reshape((-1,1))
    p1 = -F1*T1/m
    p2 = -F1*ts/m + F2*(T2-ts)/m
    p = np.concatenate((p1,p2),axis = 0)
    p = p.reshape((-1,1))
    psi = np.zeros((len(T),1))
    X = np.concatenate((l,theta,p,psi),axis = 1)
    UF1 = -F1*np.ones((len(T1),1))
    UF2 = F1*np.ones((len(T2),1))
    UF = np.concatenate((UF1,UF2),axis = 0)/length_torque_constant
    UT = np.zeros((len(T),1))
    U = np.concatenate((UF,UT),axis = 1)
    return X,U
def angle_traj(T,l0,theta0, thetaf):
    tf = T[-1]
    n = len(T)
    ns = math.ceil(n/2)
    ts = T[ns-1]
    T1 = T[0:ns]
    T2 = T[ns:]
    J = m_end*(l0+a+b)**2 + Jc + mboom*((a+l0)*rho/mboom)**2
    Tau = (thetaf - theta0)*4*J/(tf**2)
    theta1 = Tau*T1**2/(2*J) + theta0
    theta2 = Tau*ts**2/(2*J) + theta0 + Tau*ts*(T2-ts)/J - Tau*(T2-ts)**2/(2*J)
    theta = np.concatenate((theta1,theta2),axis = 0)
    theta = theta.reshape((-1,1))
    psi1 = Tau*T1/J
    psi2 = Tau*ts/J - Tau*(T2-ts)/J
    psi = np.concatenate((psi1,psi2),axis = 0)
    psi = psi.reshape((-1,1))
    l = l0*np.ones((len(T),1))
    p = np.zeros((len(T),1))
    X = np.concatenate((l,theta,p,psi),axis = 1)
    UF = np.zeros((len(T),1))
    UT1 = Tau*np.ones((len(T1),1))
    UT2 = -Tau*np.ones((len(T2),1))
    UT = np.concatenate((UT1,UT2),axis = 0)/gear_ratio
    U = np.concatenate((UF,UT),axis = 1)
    return X,U
def genTrajectory(x0,xf):
    l0 = x0[0]
    theta0 = x0[1]
    lf = xf[0]
    thetaf = xf[1]
    J = m_end*(a+b)**2 + Jc + mboom*((a)*rho/mboom)**2
    m = m_end + mboom
    dt1 = np.sqrt(4*m*(x0[0])/(max_torque*length_torque_constant))
    dt2 = np.sqrt((xf[1] - x0[1])*4*J/(max_torque*gear_ratio))
    dt3 = np.sqrt(4*m*(xf[0])/(max_torque*length_torque_constant))
    t1 = dt1
    t2 = t1+dt2
    t3 = t2+dt3
    TVec1 = np.linspace(0,t1,int(t1/dt+1))
    TVec2 = np.linspace(0,t2-t1,int((t2-t1)/dt+1))
    TVec3 = np.linspace(0,t3-t2,int((t3-t2)/dt+1))
    X1,U1 = length_traj(TVec1,l0,0,theta0,0)
    X2,U2 = angle_traj(TVec2,0,theta0,thetaf)
    X3,U3 = length_traj(TVec3,0,lf,thetaf,0)
    X = np.row_stack((X1,X2[1:][:],X3[1:][:], X3[-1][:]))
    U = np.row_stack((U1,U2[1:][:],U3[1:][:]))
    return X,U,
def extension(l0,lf, v):
    # trajectory generation for extending arm at a constant velocity
    # used primarily for grasping target object
    # v = grasping velocity
    t = (lf - l0)/v
    T = np.linspace(0,t,int(t/dt+1))
    N = T.shape[0]
    T = np.reshape(T,(N,1))
    l = np.dot(v,T)+l0
    theta = np.ones((N,1))
    psi = np.zeros((N,1))
    p = v*np.ones((N,1))
    X = np.hstack((l,theta,psi,p))
    Tau = np.zeros((N-1,1))
    Fl = np.zeros((N-1,1))
    U = np.hstack((Fl,Tau))
    return X,U,T

def on_iteration(iteration_count, xs, us, J_opt, accepted, converged):
    J_hist.append(J_opt)
    info = "converged" if converged else ("accepted" if accepted else "failed")
    print("iteration", iteration_count, info, J_opt)
    
# %% Problem Setup
dt = 0.01 
a = 13.875 # in
mc = 18.94 # lb mass of the carriage
rho = 4.85/78.74 # linear weight of boom (lb/in)
mgripper = 1 # lb
mboom = 4.85 # lb 
Jc = 562.70 # lb-in**2
b = 3.955 # offset of gripper CM from end of boom
d = 5.55 # location of gripper target
pulley_ratio = 3.82/.764
gear_ratio = 5
wheel_radius = 1.25 # 1.25" radius
length_torque_constant = pulley_ratio*wheel_radius
micro_step = 1600 # pulse per rev
max_torque = 16.82/1.25 # max_torque
boom_force = 15 # 15 lbs to retract the boom
Eb = 1.2E6 # psi
Ib = 0.36 # in**4
xi = 0.02 # damping ratio

m_end = mgripper

# Initial and target locations
offset = 2 # offset from target location. 
grasping_velocity = 1 # in/sec
x0 = np.array([36,0,0,0])
xf = np.array([36, np.pi/2,0,0])
xe = xf - np.array([offset,0,0,0])
xgrasp = xf + np.array([0,0,grasping_velocity,0])
xstart = (x0[0]+a+d)*np.cos(x0[1])
ystart = (x0[0]+a+d)*np.sin(x0[1])
xgoal = (xf[0]+a+d)*np.cos(xf[1])
ygoal = (xf[0]+a+d)*np.sin(xf[1])
# time constants for initial trajectories

# %% Generate Dynamics 
x_inputs = [
    T.dscalar("l"),
    T.dscalar("theta"),
    T.dscalar("p"),
    T.dscalar("psi"),
    ]
u_inputs = [
    T.dscalar("Fl"),
    T.dscalar("Ft"),
    ]
u_squash = tensor_constrain(u_inputs,-max_torque,max_torque)
l_temp = x_inputs[0]
p_temp = x_inputs[2]
psi_temp = x_inputs[3]
Fl_temp = u_squash[0]
Ft_temp = u_squash[1]
f = T.stack([
    x_inputs[0] + x_inputs[2]*dt,
    x_inputs[1] + x_inputs[3]*dt,
    x_inputs[2] + dt*(Fl_temp*length_torque_constant + ((a+b)*mboom*m_end + a*rho**2 + (mboom*m_end + rho**2)*l_temp)*psi_temp**2)/(mboom*m_end + rho**2),
    x_inputs[3] + dt*(mboom*Ft_temp*gear_ratio - 2*((a+b)*mboom*m_end + a*rho**2 + (mboom*m_end + rho**2)*l_temp)*p_temp*psi_temp)/(Jc*mboom + mboom*m_end*(a+b)**2 + (a*rho)**2 + l_temp*(2*(a+b)*mboom*m_end + 2*a*rho**2 + (mboom*m_end + rho**2)*l_temp))
    ])

dynamics = AutoDiffDynamics(f,x_inputs,u_inputs, hessians = True)

# %% Generate Initial Trajectories
X0,U0 = genTrajectory(x0,xe)
Xe,Ue, Te = extension(xe[0], xf[0], grasping_velocity)
N = U0.shape[0]
t = np.arange(N+1)*dt
l_og = X0[:,0]
theta_og = X0[:,1]
x_og = (l_og+a)*np.cos(theta_og)
y_og = (l_og+a)*np.sin(theta_og)
Fl_og= np.append(U0[:,0],0)
Ft_og= np.append(U0[:,1],0)

# %% Generate Costs
Q = np.diag([275,275, 275,275])
R = np.diag([.1,.1])
Q_terminal = np.diag([1000,1000,1000,1000])
cost = QRCost(Q,R, Q_terminal=Q_terminal, x_goal = xe)

#%% Optimizing Trajectory
J_hist = []
ilqr1 = iLQR(dynamics,cost,N)
xs,us = ilqr1.fit(x0,U0, on_iteration = on_iteration)
us = constrain(us,-max_torque,max_torque)

# %% Post Processing
l = xs[:,0]
theta = xs[:,1]
l_dot = xs[:,2]
theta_dot = xs[:,3]
Fl= np.append(us[:,0],0)
Ft= np.append(us[:,1],0)
x = (l+a+d)*np.cos(theta)
y = (l+a+d)*np.sin(theta)
ax = plt.figure(1)
plt.clf()
plt.plot(x,y,label="iLQR")
plt.plot(x_og,y_og,'--', label="Original")
plt.plot(xstart, ystart,'o', label="Start")
plt.plot(xgoal, ygoal,'o', label="Goal")
plt.gca().set_aspect('equal')
plt.xlim((-50,50))
plt.ylim((-10,50))
ax.legend()
plt.show()
plt.figure(2)
plt.clf()
ax1 = plt.subplot(2,1,1)
plt.title("Actuation Torque Comparison")
ax1.plot(t,Fl,label = 'iLQR')
ax1.plot(t,Fl_og,'--',label='Original')
ax1.legend()
plt.ylabel(r'$\tau_{L}$')
plt.xlabel('Time (s)')
ax2 = plt.subplot(2,1,2)
ax2.plot(t,Ft, label = 'iLQR')
ax2.plot(t,Ft_og,'--',label = 'Original')
plt.ylabel(r'$\tau_{\theta}$')
plt.xlabel('Time (s)')
ax2.legend()
        
plt.figure(3)
plt.clf()
ax2 = plt.subplot(2,1,1)
ax2.plot(t,l, label="iLQR")
ax2.plot(t,X0[:,0], label="Original")
ax2 = plt.subplot(2,1,2)
ax2.plot(t,theta, label="iLQR")
ax2.plot(t,X0[:,1], label="Original")
        
# %% Interface with Arduino
#ser = serial.Serial('COM3',115200)   
#run_trajectory(X,U,dt)
#ser.close()          
        
# %% Grasping a spinning object
m_o = 3.42 # lbs
I_o = 44.43 # lbs
r_o = 5 # 5 inches
Omega_o = 6.28 # rads/sec

N_spin = int(10/dt)   
J = mgripper*(a+b)**2 + Jc + mboom*((a)*rho/mboom)**2
R = a+b+xf[0]+r_o
Omega_f = I_o*r_o/(I_o + m_o*R**2 + J)

x0_spin = xgrasp + np.array([0,0,0,Omega_f])

m_end = mgripper + m_o
b_obj = (mgripper*b + m_o*(r_o + b))/m_end
t_spin = np.arange(N+1)*dt
U_init = np.random.uniform(-1, 1, (N, 2))

# %% New Dynamics
fspin = T.stack([
    x_inputs[0] + x_inputs[2]*dt,
    x_inputs[1] + x_inputs[3]*dt,
    x_inputs[2] + dt*(Fl_temp*length_torque_constant + ((a+b_obj)*mboom*m_end + a*rho**2 + (mboom*m_end + rho**2)*l_temp)*psi_temp**2)/(mboom*m_end + rho**2),
    x_inputs[3] + dt*(mboom*Ft_temp*gear_ratio - 2*((a+b_obj)*mboom*m_end + a*rho**2 + (mboom*m_end + rho**2)*l_temp)*p_temp*psi_temp)/(Jc*mboom + mboom*m_end*(a+b_obj)**2 + (a*rho)**2 + l_temp*(2*(a+b_obj)*mboom*m_end + 2*a*rho**2 + (mboom*m_end + rho**2)*l_temp))
    ])

dynamics_spin = AutoDiffDynamics(fspin,x_inputs,u_inputs, hessians = True)

# %% Computing Cost
Qspin = np.diag([1000,1000,1000,1000])
Rspin = np.diag([1, .05])
Qspin_terminal = np.diag([1000,1000,1000,1000])
cost_spin = QRCost(Qspin,Rspin, Q_terminal=Qspin_terminal, x_goal = xf)

#%% Optimizing Trajectory
J_hist = []
ilqr = iLQR(dynamics_spin,cost_spin,N_spin)
xs_spin,us_spin = ilqr.fit(x0_spin,U_init, on_iteration = on_iteration)
us_spin = constrain(us_spin,-max_torque,max_torque)    
        
# %% Plotting
l = xs_spin[:,0]
theta = xs_spin[:,1]
l_dot = xs_spin[:,2]
theta_dot = xs_spin[:,3]
Fl= np.append(us_spin[:,0],0)
Ft= np.append(us_spin[:,1],0)
x = (l+a+d)*np.cos(theta)
y = (l+a+d)*np.sin(theta)
ax = plt.figure(1)
plt.clf()
plt.plot(x,y,label="iLQR")
plt.plot(xgoal, ygoal,'o', label="Goal")
plt.gca().set_aspect('equal')
plt.xlim((-50,50))
plt.ylim((-10,50))
ax.legend()
plt.show()
plt.figure(2)
plt.clf()
ax1 = plt.subplot(2,1,1)
plt.title("Actuation Torque Comparison")
ax1.plot(t_spin,Fl,label = 'iLQR')
ax1.legend()
plt.ylabel(r'$\tau_{L}$')
plt.xlabel('Time (s)')
ax2 = plt.subplot(2,1,2)
ax2.plot(t_spin,Ft, label = 'iLQR')
plt.ylabel(r'$\tau_{\theta}$')
plt.xlabel('Time (s)')
ax2.legend()
        
plt.figure(3)
plt.clf()
ax2 = plt.subplot(2,1,1)
ax2.plot(t_spin,l, label="iLQR")
ax2 = plt.subplot(2,1,2)
ax2.plot(t_spin,theta, label="iLQR")   
        
# %%  Flexible Dynamics

x_flex = [
    T.dscalar("l"),
    T.dscalar("theta"),
    T.dscalar("delta"),
    T.dscalar("p"),
    T.dscalar("psi"),
    T.dscalar("vd")
    ]
l_temp = x_flex[0]
delta_temp = x_flex[2]
p_temp = x_flex[3]
psi_temp = x_flex[4]
vd_temp = x_flex[5]
Fl_temp = u_squash[0]*length_torque_constant
Ft_temp = u_squash[1]*gear_ratio
f_flex = T.stack([
    x_flex[0] + x_flex[3]*dt,
    x_flex[1] + x_flex[4]*dt,
    x_flex[2] + x_flex[5]*dt,
    x_flex[3] + dt*(2*(a**2*rho**2 + l_temp*(2*a + l_temp)*rho**2 + mboom*m_end*delta_temp**2 + Jc*mboom)*((3*b + 2*l_temp)**2*((a*rho**2 + (a + b)*mboom*m_end + (rho**2 + mboom*m_end)*l_temp)*psi_temp**2 + 2*mboom*m_end*vd_temp*psi_temp + Fl_temp*mboom)*l_temp**3 + 18*Eb*Ib*mboom*delta_temp**2*(b + l_temp)) + mboom*m_end*delta_temp*l_temp*(3*b + 2*l_temp)*(2*(3*b + 2*l_temp)*(-2*(a + l_temp)*p_temp*psi_temp*rho**2 + mboom*Ft_temp + 2*mboom*m_end*xi*(a + b + l_temp)*np.sqrt(3)*np.sqrt((Eb*Ib)/(m_end*l_temp**3))*vd_temp)*l_temp**2 + 2*mboom*delta_temp*(6*(a + b)*Eb*Ib + l_temp*(6*Eb*Ib - m_end*l_temp*(3*b + 2*l_temp)*psi_temp*(2*vd_temp + (a + b + l_temp)*psi_temp)))))/(2*l_temp**3*(3*b + 2*l_temp)**2*(mboom*m_end*rho**2*delta_temp**2 + (rho**2 + mboom*m_end)*(a**2*rho**2 + l_temp*(2*a + l_temp)*rho**2 + Jc*mboom))),
    x_flex[4] + dt*(18*Eb*Ib*mboom**2*m_end*(b + l_temp)*delta_temp**3 + mboom*l_temp*(3*b + 2*l_temp)*(6*(a + b)*Eb*Ib*(rho**2 + mboom*m_end) + l_temp*(6*Eb*Ib*(rho**2 + mboom*m_end) + m_end*l_temp*(3*b + 2*l_temp)*(Fl_temp*mboom - rho**2*psi_temp*(2*vd_temp + b*psi_temp))))*delta_temp + (rho**2 + mboom*m_end)*l_temp**3*(3*b + 2*l_temp)**2*(-2*(a + l_temp)*p_temp*psi_temp*rho**2 + mboom*Ft_temp + 2*mboom*m_end*xi*(a + b + l_temp)*np.sqrt(3)*np.sqrt((Eb*Ib)/(m_end*l_temp**3))*vd_temp))/(l_temp**3*(3*b + 2*l_temp)**2*(mboom*m_end*rho**2*delta_temp**2 + (rho**2 + mboom*m_end)*(a**2*rho**2 + l_temp*(2*a + l_temp)*rho**2 + Jc*mboom))),
    x_flex[5] + dt*-((mboom*m_end*(6*Eb*Ib*(3*b*(a + b)*mboom*m_end + l_temp*(3*b*rho**2 + 3*(a + 2*b)*mboom*m_end + (2*rho**2 + 3*mboom*m_end)*l_temp)) - m_end*rho**2*l_temp**3*(3*b + 2*l_temp)**2*psi_temp**2)*delta_temp**3 + 2*mboom*m_end**2*rho**2*l_temp**3*(3*b + 2*l_temp)**2*(xi*np.sqrt(3)*np.sqrt((Eb*Ib)/(m_end*l_temp**3))*vd_temp + p_temp*psi_temp)*delta_temp**2 - l_temp*(3*b + 2*l_temp)*(l_temp*(l_temp*(-3*b**2*Fl_temp*mboom**2*m_end**2 - 3*a*b*Fl_temp*mboom**2*m_end**2 + ((-Fl_temp)*m_end*l_temp*(2*a + 5*b + 2*l_temp)*mboom**2 + 2*m_end*rho**2*(a + b + l_temp)*(3*b + 2*l_temp)*vd_temp*psi_temp*mboom + (3*b + 2*l_temp)*(a**2*rho**4 + mboom*(Jc + (a**2 + b*a + b**2)*m_end)*rho**2 + l_temp*(b*mboom*m_end + 2*a*(rho**2 + mboom*m_end) + (rho**2 + mboom*m_end)*l_temp)*rho**2 + Jc*mboom**2*m_end)*psi_temp**2)*m_end - 6*Eb*Ib*(rho**2 + mboom*m_end)**2) - 12*Eb*Ib*(rho**2 + mboom*m_end)*(a*rho**2 + (a + b)*mboom*m_end)) - 6*Eb*Ib*(rho**2 + mboom*m_end)*(mboom*m_end*(a + b)**2 + a**2*rho**2 + Jc*mboom))*delta_temp + m_end*(rho**2 + mboom*m_end)*l_temp**3*(3*b + 2*l_temp)**2*(2*xi*(rho**2 + mboom*m_end)*np.sqrt(3)*np.sqrt((Eb*Ib)/(m_end*l_temp**3))*vd_temp*l_temp**2 + (-2*b*p_temp*psi_temp*rho**2 + mboom*Ft_temp + 4*xi*(a*rho**2 + (a + b)*mboom*m_end)*np.sqrt(3)*np.sqrt((Eb*Ib)/(m_end*l_temp**3))*vd_temp)*l_temp + (a + b)*mboom*Ft_temp + 2*xi*(a**2*rho**2 + mboom*(m_end*(a + b)**2 + Jc))*np.sqrt(3)*np.sqrt((Eb*Ib)/(m_end*l_temp**3))*vd_temp + 2*(Jc*mboom - a*b*rho**2)*p_temp*psi_temp))/(m_end*l_temp**3*(3*b + 2*l_temp)**2*(mboom*m_end*rho**2*delta_temp**2 + (rho**2 + mboom*m_end)*(a**2*rho**2 + l_temp*(2*a + l_temp)*rho**2 + Jc*mboom))))
    ])
dynamics_flex = AutoDiffDynamics(f_flex,x_flex,u_inputs, hessians = False)      
 
#%% Flexible Cost  
x0_flex = np.hstack((x0[0:2],0,x0[2:],0))
xe_flex = np.hstack((xe[0:2],0,xe[2:],0))     
Qflex = np.diag([275,275,275, 275,275, 275])
Rflex = np.diag([.1,.1])
Qflex_terminal = np.diag([1000,1000,1000,1000,1000, 1000])
cost_flex = QRCost(Qflex,Rflex, Q_terminal=Qflex, x_goal = xe_flex)   

#%% Flexible Optimization
J_hist = []
U_init = np.random.uniform(-1, 1, (N, 2))
ilqr = iLQR(dynamics_flex,cost_flex,N)
xs_flex,us_flex = ilqr.fit(x0_flex,U_init, on_iteration = on_iteration)
us_flex = constrain(us_flex,-max_torque,max_torque)   
        
#%% Plotting Flex
        
l = xs_flex[:,0]
theta = xs_flex[:,1]
delta = xs_flex[:,2]
l_dot = xs_flex[:,3]
theta_dot = xs_flex[:,4]
delta_dot = xs_flex[:,5]
Fl= np.append(us_flex[:,0],0)
Ft= np.append(us_flex[:,1],0)
x = (l+a+d)*np.cos(theta) - delta*np.sin(theta)
y = (l+a+d)*np.sin(theta) + delta*np.cos(theta)
ax = plt.figure(1)
plt.clf()
plt.plot(x,y,label="iLQR")
plt.plot(xgoal, ygoal,'o', label="Goal")
plt.gca().set_aspect('equal')
ax.legend()
plt.show()
plt.figure(2)
plt.clf()
ax1 = plt.subplot(2,1,1)
plt.title("Actuation Torque Comparison")
ax1.plot(t,Fl,label = 'iLQR')
ax1.legend()
plt.ylabel(r'$\tau_{L}$')
plt.xlabel('Time (s)')
ax2 = plt.subplot(2,1,2)
ax2.plot(t,Ft, label = 'iLQR')
plt.ylabel(r'$\tau_{\theta}$')
plt.xlabel('Time (s)')
ax2.legend()
        
plt.figure(3)
plt.clf()
ax2 = plt.subplot(2,1,1)
ax2.plot(t,l, label="iLQR")
ax2 = plt.subplot(2,1,2)
ax2.plot(t,theta, label="iLQR")     

plt.figure(4)
plt.clf()
plt.plot(t,delta)
        