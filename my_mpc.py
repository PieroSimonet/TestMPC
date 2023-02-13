# General import 

import numpy as np
import scipy.sparse as sparse
from scipy.integrate import ode
import time
import matplotlib.pyplot as plt

# pyMPC import

from pyMPC.mpc import MPCController

pos_value = [-0.2,-0.1, 0, 0.1,0.2]
#pos_value = [-0.1, 0, 0.1]


def tranform_to_sys(theta,d,u):
    Tr = np.array([[np.cos(theta) , np.sin(theta)],
                  [-np.sin(theta)/d , np.cos(theta)/d]])
    out = Tr.dot(u)
        
    if( d < 0.02 ):
        out[0] = 0
        out[1] = 0
    # discretize and campionate out
    out[0] = min(pos_value, key=lambda x:abs(x-out[0]))
    out[1] = min(pos_value, key=lambda x:abs(x-out[1]))

    realU = np.linalg.inv(Tr).dot(out)
    
    return out,realU


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

# Variables
Hz = 16
Ts = 1/Hz

A = np.eye(2)
B = Ts * np.eye(2)

x0 = np.array([0,0])

xref = np.array([1,0.4])

uminus1 = uref = np.array([0.0,0.0])

# Constraints
xmin = np.array([-4.0, -4])
xmax = np.array([4.0,   4.0])

umin = np.array([-0.1,-0.1])
umax = np.array([0.1,0.1])

Dumin = np.array([-0.1,-0.1])
Dumax = np.array([0.04,0.04])

Qx =  8 * sparse.eye(2)
QxN = 2 * sparse.eye(2)
Qu = 0.1 * sparse.eye(2)
QDu = 4 * sparse.eye(2)

Np = 80

# Initialize and setup MPC controller

K = MPCController(A,B,Np=Np, x0=x0,xref=xref,uminus1=uminus1,
                  Qx=Qx, QxN=QxN, Qu=Qu,QDu=QDu,
                  xmin=xmin,xmax=xmax,umin=umin,umax=umax)#,Dumin=Dumin,Dumax=Dumax)
K.setup() # this initializes the QP problem for the first step


len_sim = 100  # simulation length (s)
nsim = int(len_sim/Ts) # simulation length(timesteps)

[nx, nu] = B.shape # number of states and number or inputs

xsim = np.zeros((nsim,3))
xsimv = np.zeros((nsim,2))
command = np.zeros((nsim,2))
usim = np.zeros((nsim,nu))
tsim = np.arange(0,nsim)*Ts

xstep = x0
uMPC = uminus1

t0 = 0

system_dyn = ode(f_ODE).set_integrator('vode', method='bdf')
system_dyn.set_initial_value([0,0,0], t0)
system_dyn.set_f_params([0.0,0.0])




t_step = t0

for i in range(nsim):
    xsim[i,:] = system_dyn.y
    xsimv[i,:] = xstep

    #time_start = time.time_ns()
    
    pose = np.array([system_dyn.y[0],system_dyn.y[1]]) 
    
    K.update(pose,uMPC) # update with measurement
    uMPC = K.output() # MPC step (u_k value)
    
    dist = np.linalg.norm(pose-xref)
    u = tranform_to_sys(system_dyn.y[2],dist,uMPC)

    xstep = A.dot(pose) + B.dot(u[1])
    
    usim[i,:] = u[1]
    command[i,:] = u[0]
    system_dyn.set_f_params(u[0]) # set current input value
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





