import numpy as np
import do_mpc
from casadi import *
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.integrate import ode


# Customizing Matplotlib:
mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

model_type = 'continuous' #'continuous' 
model = do_mpc.model.Model(model_type)

x= model.set_variable(var_type='_x', var_name='x', shape=(1,1))
y = model.set_variable(var_type='_x', var_name='y', shape=(1,1))
theta = model.set_variable(var_type='_x', var_name='theta', shape=(1,1))

v = model.set_variable(var_type='_u', var_name='v')
w = model.set_variable(var_type='_u', var_name='w')

model.set_rhs('theta', w)

f_1 = vertcat( 
  v * cos(theta)
)

model.set_rhs('x', f_1)

f_2 = vertcat( 
  v * sin(theta)
)

model.set_rhs('y', f_2)

model.setup()

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 80,
    't_step': 0.0625,
    'n_robust': 1,
    'store_full_solution': False,
}
mpc.set_param(**setup_mpc)

xref = np.array([1,1,0]) 

mterm = (x-xref[0])**2 + (y-xref[1])**2 + (theta-xref[2])**2
lterm =  (x-xref[0])**2 + (y-xref[1])**2 + (theta-xref[2])**2

mpc.set_objective(mterm=mterm, lterm=lterm)

mpc.set_rterm(
    v=1e-2,
    w=1e-2
)

# Lower bounds on states:
mpc.bounds['lower','_x', 'x'] = -2*np.pi
mpc.bounds['lower','_x', 'y'] = -2*np.pi
mpc.bounds['lower','_x', 'theta'] = -2*np.pi
# Upper bounds on states
mpc.bounds['upper','_x', 'x'] = 2*np.pi
mpc.bounds['upper','_x', 'y'] = 2*np.pi
mpc.bounds['upper','_x', 'theta'] = 2*np.pi

# Lower bounds on inputs:
mpc.bounds['lower','_u', 'v'] = -2*np.pi
mpc.bounds['lower','_u', 'w'] = -2*np.pi
# Lower bounds on inputs:
mpc.bounds['upper','_u', 'v'] = 2*np.pi
mpc.bounds['upper','_u', 'w'] = 2*np.pi

mpc.setup()

simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step = 0.0625)

simulator.setup()

x0 = np.pi*np.array([0,0,0]).reshape(-1,1)
simulator.x0 = x0
mpc.x0 = x0
mpc.set_initial_guess()

mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)



# My car
def f_ODE(t,x,u):
    v = u[0]
    w = u[1]
    theta = x[2]

    der = np.zeros(3)
    der[0] = v * np.cos(theta)
    der[1] = v * np.sin(theta)
    der[2] = w
    return der

Ts = 0.0625

len_sim = 100  # simulation length (s)
nsim = int(len_sim/Ts) # simulation length(timesteps)

[nx, nu] = [2,2]

xsim = np.zeros((nsim,3))
xsimv = np.zeros((nsim,2))
command = np.zeros((nsim,2))
usim = np.zeros((nsim,nu))
tsim = np.arange(0,nsim)*Ts

xstep = x0

t0 = 0

system_dyn = ode(f_ODE).set_integrator('vode', method='bdf')
system_dyn.set_initial_value([0,0,0], t0)
system_dyn.set_f_params([0.0,0.0])


t_step = t0

for i in range(nsim):
    xsim[i,:] = system_dyn.y

    #time_start = time.time_ns()
    
    u = mpc.make_step(system_dyn.y)
    u = np.array(u).reshape(-1)
    usim[i,:] = u
    command[i,:] = u
    system_dyn.set_f_params(u) # set current input value
    #print("u = "+str(u));
    system_dyn.integrate(t_step + Ts)

    
    t_step += Ts

    #   print("X= "+str(xstep))

# Plot results

d = 2

fig,axes = plt.subplots(2,2, figsize=(10,10))
axes[0,0].plot(xsim[:,0], xsim[:,1], "k")
axes[0,0].plot(xref[0],xref[1],"ro")
axes[0,0].set_xlim([-d,d])
axes[0,0].set_ylim([-d,d])
axes[0,0].grid(True)
axes[0,0].set_title("Position simulation (x,y)")

axes[1,0].plot(xsimv[:,0], xsimv[:,1], "k")
axes[1,0].plot(xref[0],xref[1],"ro")
axes[1,0].set_xlim([-d,d])
axes[1,0].set_ylim([-d,d])
axes[1,0].grid(True)
axes[1,0].set_title("Position linearization (x,y)")


axes[0,1].plot(tsim, xsim[:,2])
axes[0,1].set_title("Position (theta)")
axes[0,1].grid(True)



axes[1,1].plot(tsim, command[:,0], label="v")
axes[1,1].plot(tsim, command[:,1], label="w")
axes[1,1].set_title("command (v)")
axes[1,1].grid(True)

#for ax in axes:
#    ax.grid(True)
#    ax.legend()

plt.show()









