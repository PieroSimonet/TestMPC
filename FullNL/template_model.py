from casadi import *
import do_mpc
import numpy as np

def template_model():
    # Create the model needed to evaluate mpc controller
    model_type = "continuous"
    model = do_mpc.model.Model(model_type)

    ## The state variables for now it is:
    #
    # x(t)     = v(t)cos(theta)
    # y(t)     = v(t)sen(theta)
    # theta(t) = w(t)
    #
    # with v,w considered as input to the system

    x     = model.set_variable(var_type='_x', var_name='x', shape=(1,1))
    y     = model.set_variable(var_type='_x', var_name='y', shape=(1,1))
    theta = model.set_variable(var_type='_x', var_name='theta', shape=(1,1))

    v     = model.set_variable(var_type='_u', var_name='v')
    w     = model.set_variable(var_type='_u', var_name='w')

    # Setup the variables update function

    f_1 = vertcat( v * cos(theta) )
    f_2 = vertcat( v * sin(theta) )

    model.set_rhs('x', f_1)
    model.set_rhs('y', f_2)
    model.set_rhs('theta', w)

    # Setup and return the model
    model.setup()

    return model

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
    
    

    
