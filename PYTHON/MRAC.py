import boom
import numpy as np
boom = boom.boom(2.0,0.05, 0.05, .05)
boom.eps = 1e-4
params_estimate, P = boom.get_estimate(np.array([1.0, np.pi/2, 0.0, 0.0]), np.array([1.0, 0.0 , 0.0]))