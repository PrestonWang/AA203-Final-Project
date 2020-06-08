# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:50:42 2020

@author: Preston Wang
"""

import numpy as np
import scipy as sp
from scipy import integrate
import boom
import matplotlib.pyplot as plt

# time scale
dt = 0.05 # seconds
gamma = 10*np.eye(4)
t0 = 0
tf = 2
teval = np.arange(t0,tf+dt,dt)

# size of state and control
num_state = 4
num_control = 2
num_params = 3

## Actual Model Parameters
m = 2 # kg
a = .05 # m 
b = .05 # m

# Initial Guess for Model Parameters
mref = 1 # kg
aref = 0 # m
bref = 0 # m

# Initial Location
q0 = np.array([1, np.pi/4,0,0])
thetad0 = np.array([mref, aref, bref])
y0 = np.concatenate((q0, q0, thetad0))
qprev = np.reshape(q0, (-1,1))

# indices
index_qref = [0, num_state]
index_q = [num_state, num_state*2]
index_thetad = [index_q[1], index_q[1]+num_params]

# gains:
Kv = 0.1*np.eye(2)
Kp = 0.1*np.eye(2)
Ld = 0.1*np.eye(3)
alpha = 0.5

# Goal position
xd = np.array([1.1, .9])
xd = np.reshape(xd,(-1,1))
xddot = np.array([0,0])
xddot = np.reshape(xddot, (-1,1))

def Yd_matrix(q, qdot, J, m,a,b):
    q= q.flatten()
    qdot = qdot.flatten()
    l = q[0]
    theta = q[1]
    ldot = q[2]
    thetadot = q[3]
    lddot = qdot[0]
    thetaddot = qdot[1]
    return np.array([[m*l*thetadot**2 - m*lddot, m*thetadot**2, m*thetaddot],[(J/m + (a**2 + b**2) + l**2)*thetadot + 2*m*l*ldot*thetadot, 2*m*l*thetadot + 2*m*ldot*thetadot, -m*lddot]])
def mrac_ode(t,y):
    # MRAC odes for use with scipy solve_ivp
    # t is a scalar
    # y is shape (4n)
    global qprev 
    qref = np.reshape(y[index_qref[0]:index_qref[1]],(num_state,1))
    q = np.reshape(y[index_q[0]:index_q[1]],(num_state,1))
    thetad = np.reshape(y[index_thetad[0]: index_thetad[1]],(num_params,1))
    m = thetad[0,:]
    a = thetad[1,:]
    b = thetad[2,:]
    Jq = boom.Jq(q)
    x = boom.forward_kinematics(q)
    deltax = x-xd
    xdot_hat = boom.Jq(qref) @ qref[0:2,:]
    deltaxdot = xdot_hat - xddot
    qdot = (qref - qprev)/dt
    Yd = Yd_matrix(q,qdot,boom.J,mref,aref,bref)
    u = -Jq @ (Kv @ deltaxdot + Kp @ deltax) + Yd@thetad
    dqref = boom.dynamics(qref,u,mref,aref,bref)
    dq = boom.dynamics(q,u,m,a,b)
    dq.flatten()
    s = np.linalg.inv(Jq) @ (deltaxdot + alpha*deltax)
    dtheta = -Ld @ np.transpose(Yd) @ s
    dy = np.vstack((dqref, dq, dtheta))
    dy = dy.flatten()
    qprev = qref
    return dy

sol = sp.integrate.solve_ivp(mrac_ode,(t0,tf),y0, t_eval = teval)
Y = sol.y
qref_sol = Y[index_qref[0]:index_qref[1],:]
q_sol = Y[index_q[0]:index_q[1],:]
thetad_sol = Y[index_thetad[0]: index_thetad[1]]
e_sol = qref_sol - q_sol
plt.figure(1)
plt.clf()
plt.plot(sol.t, qref_sol[0,:], sol.t, q_sol[0,:])