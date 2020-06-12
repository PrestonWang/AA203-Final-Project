import boom
import numpy as np
boom = boom.boom(25,0.5, 0.1, .07)
boom.eps = 1e-6
q0 = np.array([5.0, np.pi/2,0.0,0.0])# initial state
params_estimate, P = boom.get_estimate(q0, np.array([10, 1.5/2, 0]))


