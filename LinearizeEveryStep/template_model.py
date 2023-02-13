import numpy as np

# The full model 
def f_ODE(t,x,u):
    v = u[0]
    w = u[1]
    theta = x[2]

    ## The state variables for now it is:
    #
    # x(t)     = v(t)cos(theta)
    # y(t)     = v(t)sen(theta)
    # theta(t) = w(t)
    #
    # with v,w considered as input to the system
    

    der = np.zeros(3)
    der[0] = v * np.cos(theta)
    der[1] = v * np.sin(theta)
    der[2] = w
    return der

## The expanded model is
# Linearization near x0
# f_l(x) = f(x0) + Df(x0)(x-x0)
def get_AB_linearized_extended(x):
    ## The state variables is in the form
    # x = [x,y,theta,v,w]
    # u = [av, ar]

    A = np.array([[ 0, 0, -x[3] * np.sin(x[2]), np.cos(x[2]), 0],
                  [ 0, 0,  x[3] * np.cos(x[2]), np.sin(x[2]), 0],
                  [ 0, 0,  0, 0, 1],
                  [ 0, 0,  0, 0, 0],
                  [ 0, 0,  0, 0, 0]])
    B = np.array([[0,0],
                  [0,0],
                  [0,0],
                  [1,0],
                  [0,1]])
                  
    return A,B


