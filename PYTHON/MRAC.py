# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 14:50:42 2020

@author: Preston Wang
"""

import numpy as np
import scipy as sp
import boom

# time scale
dt = 0.05 # seconds
gamma = 2*np.eye(2)
t0 = 0
tf = 5

# size of state and control
num_state = 4
num_control = 2

## Actual Model Parameters
m = 1 # kg
a = .05 # m 
b = .05 # m

# Initial Guess for Model Parameters
mref = 2 # kg
aref = 0 # m
bref = 0 # m

# Initial Location
x0 = np.array([1, np.pi/4,0,0])
u0 = np.array([0,0])
kr0 = np.zeros((num_state,))
kx0 = np.zeros((num_state,))

# getting linearized matrices
Aref, Bref = boom.linearize_dynamics(x0, u0, mref, aref, bref)
A, B = boom.linearize_dynamics(x0, x0, m, a, b)

def mrac_ode(t,y):
    # MRAC odes for use with scipy solve_ivp
    # t is a scalar
    # y is shape (4n)
    xref = np.reshape(y[0:num_state],(-1,1))
    x = np.reshape(y[num_state:num_state*2],(-1,1))
    kr = np.reshape(y[num_state*2: num_state*3],(-1,1))
    kx = np.reshape(y[num_state*3: num_state*4],(-1,1))
    
    r = np.array([0.1*np.sin(t),0])
    e = x-xref
    dxref = x0 + dt*Aref @ (xref-x0) + dt * Bref @ (r - u0)
    dx = x0 + dt*(A - B@kx) @ (x-x0) + dt*B @ (kr@r - u0)
    dkr = -np.sign(B)@gamma@r@np.transpose(e)
    dkx = -np.sign(B)@gamma@r@np.transpose(e)
    
#sp.integrate.solve_ivp(mrac_ode(t,y),(t0,tf), )
    