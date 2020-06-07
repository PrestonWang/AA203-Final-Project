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
t0 = 0
tf = 10
dt = 0.05
teval = np.arange(t0,tf+dt,dt)

# size of state and control
num_state = 4
num_control = 2

## Actual Model Parameters
m = 1 # kg
a = 0.25 # m 
b = 0.25 # m

# centroid locations:
ap = 0.1
bp = 0

# Initial Location
q0 = np.array([[1], [np.pi/4],[0],[0]])
x0 = boom.forward_kinematics(q0,a,b)
u0 = np.zeros((num_control,1))
def u_probe(t):
    return np.array([0.01*(np.sin(np.pi*t)), 0.01*(np.cos(np.pi*t))])
def mrac_ode(t,y):
    u = u_probe(t)
    dy = boom.dynamics(y,u,m,a,b).flatten()
    return dy

sol = sp.integrate.solve_ivp(mrac_ode,(t0,tf),q0.flatten(), t_eval = teval)
qsol = sol.y

# getting mass location
Xsol = np.zeros((num_state,qsol.shape[1]))
Usol = np.zeros((num_control, qsol.shape[1]))
Upsilon = Usol
for i in range(qsol.shape[1]):
    q = qsol[:,i]
    Xsol[0:2,i] = boom.forward_kinematics(q, ap, bp).flatten()
    Jq = boom.Jq(q,ap,bp)
    Xsol[2:4,i] = (Jq@ np.array([[q[2]],[q[3]]])).flatten()
    Usol[:,i] = u_probe(sol.t[i])
    Upsilon[:,i] = np.dot(Jq, u_probe(sol.t[i]))
Upsilon = Usol[:,0:-1]
X = Xsol[:,0:-1]
Xp = Xsol[:,1:]
Omega = np.vstack((X,Upsilon))
U_tilda, Sigma_tilda, V_tilda = np.linalg.svd(Omega)
U_hat, Sigma_hat, V_hat = np.linalg.svd(Xp)
S_tilda = sp.linalg.diagsvd(Sigma_tilda, num_state + num_control, qsol.shape[1]-1)
U1 = U_tilda[:,0:num_state]
U2 = U_tilda[:,num_state:]
Sinv = np.linalg.pinv(S_tilda)
Atilda = U_hat @ Xp @ V_tilda @ Sinv @ U1 @ U_hat
#%% plotting
plt.figure(1)
plt.clf()
plt.plot(Xsol[0,:], Xsol[1,:])
