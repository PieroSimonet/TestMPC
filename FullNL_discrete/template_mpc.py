import do_mpc
import numpy as np
from casadi import *

distance_xy    = 200
distance_theta = 2 * np.pi

vel_v          = 0.2
vel_w          = 0.1

verbose = False

Qx     = np.array([10,10,1e-3])
QxN    = np.array([4,4,1e-3])

R      = np.array([1e-2,1e-1]) 

def template_mpc(model, xref, Ts):
    # Obtain an instance of the do-mpc MPC class and initiate it with the model
    mpc = do_mpc.controller.MPC(model)

    # Setup parameters
    setup_mpc = {
        'n_horizon': 40,
        't_step': Ts,
        'store_full_solution': False,
    }
    mpc.set_param(**setup_mpc)
    # mpc.set_param(nlpsol_opts = {'ipopt.linear_solver': 'pardiso'})

    if( not verbose ):
        # Soppres the output of the controller 
        suppress_ipopt = {'ipopt.print_level':0, 'ipopt.sb': 'yes', 'print_time':0}
        mpc.set_param(nlpsol_opts = suppress_ipopt)

    # Setup the cost function
    mterm = Qx[0] * (model._x['x'] - xref[0])**2 + Qx[1] * (model._x['y'] - xref[1])**2 + Qx[2] * (model._x['theta'] - xref[2])**2
    lterm = QxN[0] * (model._x['x'] - xref[0])**2 + QxN[1] * (model._x['y'] - xref[1])**2 + QxN[2] * (model._x['theta'] - xref[2])**2

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm( v=R[0], w=R[1] )

    # Lower bounds on states:
    mpc.bounds['lower','_x', 'x']     = - distance_xy
    mpc.bounds['lower','_x', 'y']     = - distance_xy
    mpc.bounds['lower','_x', 'theta'] = - distance_theta
    # Upper bounds on states
    mpc.bounds['upper','_x', 'x']     = distance_xy
    mpc.bounds['upper','_x', 'y']     = distance_xy
    mpc.bounds['upper','_x', 'theta'] = distance_theta

    # Lower bounds on inputs:
    mpc.bounds['lower','_u', 'v']     = - vel_v
    mpc.bounds['lower','_u', 'w']     = - vel_w
    # Lower bounds on inputs:
    mpc.bounds['upper','_u', 'v']     = vel_v
    mpc.bounds['upper','_u', 'w']     = vel_w

    # Setup and return the mpc controller
    mpc.setup()

    return mpc


