from pyMPC.mpc import MPCController
import numpy as np


#xmin = np.array([-4.0, -4])
#xmax = np.array([4.0,   4.0])
#umin = np.array([-0.1,-0.1])
#umax = np.array([0.1,0.1])
#Dumin = np.array([-0.1,-0.1])
#Dumax = np.array([0.04,0.04])

Qx =  8 * sparse.eye(5)
QxN = 2 * sparse.eye(5)
Qu = 0.1 * sparse.eye(2)
QDu = 4 * sparse.eye(2)

uminus1 = np.array([0,0])

def template_mpc(A,B,x0,xref):

    K = MPCController(A,B,Np=Np, x0=x0,xref=xref,uminus1=uminus1,
                      Qx=Qx, QxN=QxN, Qu=Qu,QDu=QDu) #,
                      #xmin=xmin,xmax=xmax,umin=umin,umax=umax)
    K.setup()

    return K
