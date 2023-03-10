import do_mpc

def template_simulation(model, Ts):
    # Obtain an instance of the do-mpc simulator class
    # and initiate it with the model:
    simulator = do_mpc.simulator.Simulator(model)

    # Set parameter(s):
    simulator.set_param(t_step = Ts)

    # Optional: Set function for parameters and time-varying parameters.

    # Setup and return the simulator:
    simulator.setup()

    return simulator

