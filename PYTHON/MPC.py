import numpy as np
import boom
import do_mpc
from casadi import *
from casadi.tools import *
import matplotlib.pyplot as plt
import matplotlib as mpl


# setting up boom model

# actual model values
m = 2.0 # end mass
a = 0.05 # location of center of mass x
b = 0.05 # location of center of mass y
J = .07 # kg-m**3

# setting up initial and target states
# q = [l, theta, l', theta']
q0 = np.array([1.0, np.pi/2,0.0,0.0])# initial state
qd = np.array([2.0, 0.0, 0.0, 0.0]) # target state

# LQR costs matrices
Q = np.array([275.0,275.0,275.0,275.0])
Qn = np.array([275.0,275.0,275.0,275.0]) # terminal cost
R = np.array([1.0,1.0]) # control cost

# creating boom class and finding initial estiamtes of Boom
Boom = boom.boom(m,a,b,J)
params_guess = np.array([1.0, 0.0, 0.0])
params_est, params_cov = Boom.get_estimate(q0, params_guess)

# Length of Simulation
T = 100 # 60 iterations
dt = 0.05 # time step

#%% Setting up MPC
model_type = 'continuous' # either 'discrete' or 'continuous'
model = do_mpc.model.Model(model_type)
# setting up states
l = model.set_variable(var_type = '_x', var_name = 'l', shape=(1,1))
theta = model.set_variable(var_type = '_x', var_name = 'theta', shape=(1,1))
lp = model.set_variable(var_type = '_x', var_name = 'lp', shape=(1,1))
thetap = model.set_variable(var_type = '_x', var_name = 'thetap', shape=(1,1))
# setting up inputs
Fl = model.set_variable(var_type = '_u', var_name = 'Fl', shape=(1,1))
Ft = model.set_variable(var_type = '_u', var_name = 'Ft', shape=(1,1))
# setting up parameters
m_e = model.set_variable('parameter','m_e')
a_e = model.set_variable('parameter','a_e')
b_e = model.set_variable('parameter','b_e')
# setting rhs side of equation for the dynamics (nonlinear dynamics)
model.set_rhs('l', lp)
model.set_rhs('theta', thetap)
Jp = J + m_e*a_e**2 + m_e*l*(2*a_e + l)
model.set_rhs('lp', -(b_e*m_e*(Ft - 2*(a_e + l)*lp*m_e*thetap) - (-J - (a_e**2 + b_e**2)*m_e - l*(2*a_e + l)*m_e)*(Fl + (a_e + l)*m_e*thetap**2))/(-m_e*Jp))
model.set_rhs('thetap', -(((-b_e)*Fl - Ft + 2*a_e*lp*m_e*thetap + 2*l*lp*m_e*thetap - a_e*b_e*m_e*thetap**2 - b_e*l*m_e*thetap**2)/(Jp)))

# setting up useful expressions (converting from cylindrical to cartesian coordinates)
model.set_expression('x_end', l* cos(theta))
model.set_expression('y_end', l*sin(theta))

# setting up the model
model.setup()
mpc = do_mpc.controller.MPC(model)
setup_mpc = {
    'n_horizon': 20,
    't_step': dt,
    'n_robust': 1,
    'store_full_solution': True,
}
mpc.set_param(**setup_mpc)

# setting up the objective
state_cost = Q[0]*(l-qd[0])**2 + Q[1]*(theta-qd[1])**2 + Q[2]*(lp-qd[2])**2 + Q[3]*(thetap-qd[3])**2
final_cost = Qn[0]*(l-qd[0])**2 + Qn[1]*(theta-qd[1])**2 + Qn[2]*(lp-qd[2])**2 + Qn[3]*(thetap-qd[3])**2
mpc.set_objective(mterm = final_cost, lterm = state_cost)
mpc.set_rterm(Fl = R[0], Ft = R[1])

# setting up constraint
mpc.bounds['lower', '_u', 'Fl'] = -10 # N-m
mpc.bounds['upper','_u', 'Fl'] = 10 # N-m
mpc.bounds['lower','_u', 'Ft'] = -10 #N-m
mpc.bounds['upper', '_u', 'Ft'] = 10 # N-m

#TODO: Add in Nonlinear constraints for location of end effector (make sure it can't hit any walls)
# mpc.set_nl_cons('cons_name', expression, upper_bound, soft_constraint=False)
m_e_value = params_est[0]
a_e_value = params_est[1]
b_e_value = params_est[2]
std = np.sqrt(np.diag(params_cov))
m_range = [m_e_value, m_e_value - std[0], m_e_value + std[0]]
a_range = [a_e_value, a_e_value - std[1], a_e_value + std[1]]
b_range = [b_e_value, b_e_value - std[2], b_e_value + std[2]]
mpc.set_uncertainty_values(
    m_e = m_range,
    a_e = a_range,
    b_e = b_range)
mpc.setup()

#%% configuring the simulator #%%
simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = 0.1)
p_template = simulator.get_p_template()
def p_fun(t_now):
    p_template['m_e'] = m_e_value
    p_template['a_e'] = a_e_value
    p_template['b_e'] = b_e_value
    return p_template

simulator.set_p_fun(p_fun)
simulator.setup()
q0 = q0.reshape(-1,1) # to get it into a column vector
simulator.set_initial_state(q0, reset_history=True)
mpc.set_initial_state(q0, reset_history=True)

#%% Setting up Plots
mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)

# We just want to create the plot and not show it right now. This "inline magic" supresses the output.
fig, ax = plt.subplots(2, sharex=True, figsize=(16,9))
fig.align_ylabels()

for g in [sim_graphics, mpc_graphics]:
    # Plot the angle positions (phi_1, phi_2, phi_2) on the first axis:
    g.add_line(var_type='_x', var_name='l', axis=ax[0])
    g.add_line(var_type='_x', var_name='theta', axis=ax[0])

    # Plot the set motor positions (phi_m_1_set, phi_m_2_set) on the second axis:
    g.add_line(var_type='_u', var_name='Fl', axis=ax[1])
    g.add_line(var_type='_u', var_name='Ft', axis=ax[1])


ax[0].set_ylabel('')
ax[1].set_ylabel('Motor Torques [Nm]')
ax[1].set_xlabel('time [s]')

#%% Running Simulation
for i in range(T):
    u0 = mpc.make_step(q0)
    q0 = simulator.make_step(u0)

#%% Plotting results
mpc_graphics.plot_predictions(t_ind=0)
# Plot results until current time
sim_graphics.plot_results()
sim_graphics.reset_axes()
fig
plt.show()
#TODO: Add in animations
#TODO: Add in plots of trajectory with model dispersion
