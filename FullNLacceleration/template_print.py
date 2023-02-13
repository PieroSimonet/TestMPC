import matplotlib.pyplot as plt
import matplotlib as mpl

d = 2

def my_plot(xsim,command,tsim,xref):

    fig,axes = plt.subplots(2,2, figsize=(10,10))
    axes[0,0].plot(xsim[:,0], xsim[:,1], "k")
    axes[0,0].plot(xref[0],xref[1],"ro")
    axes[0,0].set_xlim([-d,d])
    axes[0,0].set_ylim([-d,d])
    axes[0,0].grid(True)
    axes[0,0].set_title("Position simulation (x,y)")

    axes[1,0].plot(tsim, xsim[:,2])
    axes[1,0].set_title("Position (theta)")
    axes[1,0].grid(True)

    axes[0,1].plot(tsim, command[:,0], label="v")
    axes[0,1].set_title("command (v)")
    axes[0,1].grid(True)

    axes[1,1].plot(tsim, command[:,1], label="w")
    axes[1,1].set_title("command (w)")
    axes[1,1].grid(True)

    plt.show()

