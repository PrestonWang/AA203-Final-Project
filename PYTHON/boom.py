# -*- coding: utf-8 -*-
"""
Setting up iLQR to get controls for intial problem. 

@author: Preston Wang
"""
import numpy as np
dt = 0.05 # seconds
J = .07 # kg-m**3

def linearize_dynamics(x,u,m,a,b):
    l = x[0]
    theta = x[1]
    lp = x[2]
    thetap = x[3]
    Fl = u[0]
    Ft = u[1]
    Jp = J + m*a**2 + m*l*(2*a + l)
    A = np.array([
        [0, 0, 1, 0], 
        [0, 0, 0, 1],
        [(1/Jp**2)*(-2*b*(b*Fl + Ft)*(a + l)*m + 2*b*lp*m*(-J + a**2*m + l*(2*a + l)*m)*thetap + (J**2 + (2*a**2 + b**2)*J*m + a**2*(a - b)*(a + b)*m**2 + l*(2*a + l)*m*(2*J + 2*a**2*m - b**2*m + l*(2*a + l)*m))*thetap**2), 0, -((2*b*(a + l)*m*thetap)/Jp), (2*(a + l)*((-b)*lp*m + (J + (a**2 + b**2)*m + l*(2*a + l)*m)*thetap))/Jp], 
        [(m*(-2*(b*Fl + Ft)*(a + l) + 2*lp*(-J + a**2*m + l*(2*a + l)*m)*thetap + b*(J - a**2*m - l*(2*a + l)*m)*thetap**2))/Jp**2, 0, -((2*(a + l)*m*thetap)/Jp), -((2*(a + l)*m*(lp - b*thetap))/Jp)]
        ])
    B = np.array([
        [0, 0],
        [0, 0], 
        [b**2/Jp + 1/m, b/Jp], 
        [b/Jp, 1/Jp]])
    return A,B

def dynamics(x,u,m,a,b):
    l = x[0]
    theta = x[1]
    lp = x[2]
    thetap = x[3]
    Fl = u[0]
    Ft = u[1]
    Jp = J + m*a**2 + m*l*(2*a + l)
    f = np.array([
        [lp], 
        [thetap], 
        [-(b*m*(Ft - 2*(a + l)*lp*m*thetap) - (-J - (a**2 + b**2)*m - l*(2*a + l)*m)*(Fl + (a + l)*m*thetap**2))/(-m*Jp)], 
        [-(((-b)*Fl - Ft + 2*a*lp*m*thetap + 2*l*lp*m*thetap - a*b*m*thetap**2 - b*l*m*thetap**2)/(Jp))]
    ])
    return f

    
<<<<<<< Updated upstream
  
=======
def Jq(q, a,b):
    # q = joint states
    # theta_k = vector of parameters 
    q = q.flatten()
    l = q[0]
    theta = q[1]
    return np.array([[np.cos(theta), (a+l)*np.sin(theta) - b*np.cos(theta)],[np.sin(theta), (a+l)*np.cos(theta) - b*np.sin(theta)]])

def forward_kinematics(q,a,b):
    q = q.flatten()
    l = q[0]
    theta = q[1]
    return np.array([[(l+a)*np.cos(theta) - b*np.sin(theta)],[(l+a)*np.sin(theta) + b*np.cos(theta)]])
>>>>>>> Stashed changes
