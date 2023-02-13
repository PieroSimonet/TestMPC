import numpy as np
import do_mpc
import sys
import time

from template_model      import *
from template_mpc        import *
from template_simulation import *
from template_print      import my_plot

xref = np.array([1,1,0,0,0])
x0   = np.array([0,0,0,0,0])

Hz   = 16
Ts   = 1/Hz
len_sim = 100  # simulation length (s)

N_iterations = int(len_sim/Ts) # simulation length(timesteps)

model     = f_ODE()
mpc       = template_mpc(model,xref,Ts)
simulator = template_simulation(model,Ts)

simulator.x0 = x0
mpc.x0 = x0
mpc.set_initial_guess()

xsim = np.zeros((N_iterations,3))
usim = np.zeros((N_iterations,2))
tsim = np.arange(0,N_iterations)*Ts

x = x0

for k in range(N_iterations):

    time_start = time.time_ns()
    mpc       = template_mpc(model,xref,Ts)
    mpc.x0 = x
    mpc.set_initial_guess()
    u = mpc.make_step(x)
    time_end = time.time_ns()

    print("cicle " + str(k) + " time in ms = " + str( (time_end-time_start)*1e-6 ));
    
    x = simulator.make_step(u)
    
    xsim[k,:] = x.reshape(-1)
    usim[k,:] = u.reshape(-1)

my_plot(xsim,usim,tsim,xref)


