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
tf = 10
teval = np.arange(t0+dt,tf,dt)

# size of state and control
num_state = 4
num_control = 2

## Actual Model Parameters
m = 5 # kg
a = .05 # m 
b = .05 # m

# Initial Guess for Model Parameters
mref = 1 # kg
aref = 0 # m
bref = 0 # m

# Initial Location
x0 = np.array([[1], [np.pi/4],[0],[0]])
u0 = np.zeros((num_control,1))
kr0 = np.zeros((num_control,num_control)) # num_control x num_control
kx0 = np.zeros((num_control,num_state)) # num_control x num_state
y0 = np.vstack((x0, x0, np.reshape(kr0, (-1,1)), np.reshape(kx0, (-1,1))))
y0 = y0.flatten()

# indices
index_xref = [0, num_state]
index_x = [num_state, num_state*2]
index_kr = [num_state*2, num_state*2 + num_control*num_control]
index_kx = [index_kr[1], index_kr[1] + num_control*num_state]

# getting linearized matrices
Aref, Bref = boom.linearize_dynamics(x0.flatten(), u0.flatten(), mref, aref, bref)
Fref = boom.dynamics(x0.flatten(), u0.flatten(), mref, aref, bref)
A, B = boom.linearize_dynamics(x0.flatten(), u0.flatten(), m, a, b)
F = boom.dynamics(x0.flatten(), u0.flatten(), m, a, b)
Breft = np.transpose(Bref)

def r(t):
    return np.array([[5*np.sin(np.pi*t + np.pi/2)],[0]])

def mrac_ode(t,y):
    # MRAC odes for use with scipy solve_ivp
    # t is a scalar
    # y is shape (4n)
    xref = np.reshape(y[index_xref[0]:index_xref[1]],(num_state,1))
    x = np.reshape(y[index_x[0]:index_x[1]],(num_state,1))
    #kr = np.reshape(y[index_kr[0]: index_kr[1]],(num_control,num_control))
    kr = np.linalg.pinv(B) @ Bref 
    #kx = np.reshape(y[index_kx[0]: index_kx[1]],(num_control,num_state))
    kx = np.linalg.pinv(B) @ (A-Aref)
    r_t = r(t)
    e = x-xref
    dxref = F + dt*Aref @ (xref-x0) + dt * Bref @ (r_t - u0)
    dx = F + dt*(A - B@kx) @ (x-x0) + dt*B @ (kr@r_t - u0)
    dkr = -np.sign(Breft)@gamma@e@np.transpose(r_t)
    dkx = -np.sign(Breft)@gamma@e@np.transpose(x)
    dy = np.vstack((dxref, dx, np.reshape(dkr,(-1,1)), np.reshape(dkx,(-1,1))))
    dy = dy.flatten()
    return dy

sol = sp.integrate.solve_ivp(mrac_ode,(t0,tf),y0, t_eval = teval)
Y = sol.y
xref_sol = Y[index_xref[0]:index_xref[1],:]
x_sol = Y[index_x[0]:index_x[1],:]
kr_sol = Y[index_kr[0]:index_kr[1],:]
kx_sol = Y[index_kx[0]:index_kx[1],:]
e_sol = xref_sol - x_sol
plt.figure(1)
plt.clf()
plt.plot(sol.t, xref_sol[0,:], sol.t, x_sol[0,:])
plt.figure(2)
plt.clf()
plt.plot(sol.t, np.transpose(e_sol))