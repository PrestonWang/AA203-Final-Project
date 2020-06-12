# -*- coding: utf-8 -*-
"""
Setting up iLQR to get controls for intial problem. 

@author: Preston Wang
"""
# -*- coding: utf-8 -*-
import numpy as np

class boom:
    # time scale
    dt = 0.05  # seconds
    # size of state and control
    num_state = 4
    num_control = 2
    num_params = 3
    # measurement noise
    sigma = 0.01
    Sigma = np.eye(num_params)
    # RLS parameters
    max_iter = 1000 # max number of iterations
    eps = 1e-6 # termination threshold

    def __init__(self, m, a, b, J):
        # m is the actual mass of the object at the end
        # a is the actual x location of the center of mass
        # b is the actual y location of the center of mass
        # J is the moment of inertia of the base
        self.m = m
        self.b = b
        self.a = a
        self.J = J

    def get_measurements(self, q, qdot):
        # simulating getting measurements from sensors.
        # q is [l, theta, l', theta']
        # qdot is [l', theta', l'', theta''] These can be found from the dynamics equation
        # returns acceleration and force measurements with added measurement noise
        l = q[0]
        ax = qdot[2]
        ay = qdot[3] * l
        accel = np.array([ax, ay]) + 0.01*np.random.normal(0, self.sigma, (2,))  # adding measurement noise
        force = np.array([self.m * ax, self.m * ay * (l + self.a) / l,
                          self.m * self.b * ax + self.m * ay * (l + self.a)]) + np.random.normal(0, self.sigma, (3,))  # adding measurement noise
        return accel, force

    def rls(self, accel, force, q, x, P):
        # performing recursive least squares on the estimated parameters
        # accel are the acceleration measurements (1d array)
        # force are the force measurements (1d array)
        # q is the current state vector (1d array)
        # x is the current parameter estimates (column vector)
        # P is the covariance matrix
        ax = accel[0]
        ay = accel[1]
        m = x[0, 0]
        l = q[0]
        H = np.array([[ax, 0, 0], [ay, m * ay / l, 0], [ay * l, m * ay, m * ax]])
        Ht = np.transpose(H)
        K = P @ Ht @ np.linalg.inv(H @ P @ Ht + self.Sigma)
        P = (np.eye(self.num_params) - K @ H) @ P
        force = np.reshape(force, (self.num_params, 1))
        x = x + K @ (force - H @ x)
        return x, P

    def get_estimate(self, q0, x0):
        # calculate an initial estimate for the states from an excitation trajectory
        # q0 is the initial state (1d array)
        # x0 is the initial parameter estimate (1d array) [m,a,b]
        x = np.reshape(x0, (self.num_params, 1))
        q = q0
        Xe = x
        i = 0
        t = 0
        eps = self.eps
        max_iter = self.max_iter
        P = 1e8 * np.eye(self.num_params)
        while i < max_iter:
            # calculate control
            u = np.array([np.sin(np.pi * t), np.cos(np.pi * t)])
            # step forward
            f = self.dynamics(q, u, self.m, self.a, self.b)
            f = f.flatten()
            q = f * self.dt + q  # integrate forward
            # get measurements
            accel, force = self.get_measurements(q, f)
            # run recursive least squares estimate to get parameter.
            xnew, P = self.rls(accel, force, q, x, P)
            Xe = np.hstack((Xe, xnew))
            norm = np.linalg.norm(xnew - x)
            if norm < eps:
                #terminate RLS if norm of change in estimate is below a certain amount.
                break
            i += 1
            x = xnew
            t += self.dt
        return x.flatten(), P

    def dynamics(self, q, u, m, a, b):
        # calculate dynamics of arm
        # q is the current state (1d array or column vector)
        # u is the current control (1d array of column vector) [Fl, Ft]
        # m,a,b are the parameters of the end mass
        q = q.flatten()
        u = u.flatten()
        l = q[0]
        theta = q[1]
        lp = q[2]
        thetap = q[3]
        Fl = u[0]
        Ft = u[1]
        J = self.J
        Jp = J + m * a ** 2 + m * l * (2 * a + l)
        f = np.array([
            [lp],
            [thetap],
            [-(b * m * (Ft - 2 * (a + l) * lp * m * thetap) - (-J - (a ** 2 + b ** 2) * m - l * (2 * a + l) * m) * (
                    Fl + (a + l) * m * thetap ** 2)) / (-m * Jp)],
            [-(((-b) * Fl - Ft + 2 * a * lp * m * thetap + 2 * l * lp * m * thetap - a * b * m * thetap ** 2 - b * l * m * thetap ** 2) / (Jp))]
        ])
        return f

    def Jq(self, q):
        # q = joint states
        # theta_k = vector of parameters
        q = q.flatten()
        l = q[0]
        theta = q[1]
        return np.array([[np.cos(theta), -l * np.sin(theta)], [np.sin(theta), l * np.cos(theta)]])

    def forward_kinematics(self, q):
        q = q.flatten()
        l = q[0]
        theta = q[1]
        return np.array([[l * np.cos(theta)], [l * np.sin(theta)]])
