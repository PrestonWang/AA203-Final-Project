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
t0 = 0
tf = 5
teval = np.arange(t0,tf+dt,dt)

# size of state and control
num_state = 4
num_control = 2
num_params = 3

## Actual Model Parameters
m = 2.0 # kg
a = .05 # m 
b = .05 # m

# measurement noise
sigma = 0.01
Sigma = np.eye(num_params)

# RLS parameters
max_iter = 1000
eps = 1e-3
P = 1e8*np.eye(num_params)
#Setting initial values

# Initial Guess for Model Parameters
m_e = 1.0 # kg
a_e = 0.0 # m
b_e = 0.0 # m
x0 = np.array([m_e, a_e, b_e])
x0 = np.reshape(x0,(num_params,1))
# Initial Location
q0 = np.array([1.0, np.pi/4,0.0,0.0])
# Initial Values
x = x0
q= q0
Xe = x0
i = 0
t = 0

def get_measurements(q,qdot):
    l = q[0]
    ax = qdot[2]
    ay = qdot[3]*l
    accel = np.array([ax,ay]) + np.random.normal(0,sigma,(2,)) # adding measurement noise
    force = np.array([m*ax, m*ay*(l+a)/l, m*b*ax + m*ay*(l+a)]) + np.random.normal(0,sigma,(3,)) # adding measurement noise
    return accel,force

def rls(accel,force,q,x, P):
    ax = accel[0]
    ay = accel[1]
    m = x[0,0]
    l = q[0]
    H = np.array([[ax, 0,0],[ay,m*ay/l,0],[ay*l, m*ay, m*ax]])
    Ht = np.transpose(H)
    K = P@Ht@np.linalg.inv(H@P@Ht + Sigma)
    P = (np.eye(num_params) - K@H)@P
    force = np.reshape(force,(num_params,1))
    x = x + K@(force - H@x)
    return x,P


while i < max_iter:
    # calculate control
    u = np.array([np.sin(np.pi*t), np.cos(np.pi*t)])
    #step forward
    f = boom.dynamics(q, u, m,a,b)
    f = f.flatten()
    q = f*dt + q # integrate forward
    # get measurements
    accel,force = get_measurements(q,f)
    # run recursive least squares estimate to get parameter. 
    xnew,P = rls(accel,force, q,x, P)
    Xe = np.hstack((Xe,xnew))
    norm = np.linalg.norm(xnew-x)
    if norm <eps:
        break
    i += 1
    x = xnew
    t += dt

m_e = x[[0,0]]
a_e = x[[1,0]]
b_e = x[[2,0]]


#%% plotting
T = np.arange(0,t+dt,dt)
plt.figure(0)
plt.clf()
plt.plot(T,np.transpose(Xe))
plt.legend(('m','a','b'))
