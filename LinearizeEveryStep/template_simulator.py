from scipy.integrate import ode

class simulator:
    def __init__(self,f, Ts, t0 = 0):
        self.system_dyn = ode(f_ODE).set_integrator('vode', method='bdf')
        self.initialized = False
        self.Ts = Ts
        self t_step = t0
        
        
    def __init__(self,f,x0,u0,Ts,t0=0):
        self.__init__(f,Ts, t0)
        self.initialize(x0,u0,t0)

    def is_initialized(self):
        return self.initialized

    def initialize(self,x0,u0,t0=0):
        self.system_dyn.set_initial_value(x0, t0)
        self.system_dyn.set_f_params(u0)
        self t_step = t0
        self.initialized = True

    def make_step(self,u):
        if ( self.is_initialized()):
            self.system_dyn.set_f_params(u) # set current input value
            self.system_dyn.integrate(self.t_step + self.Ts)

            self.t_step += self.Ts

            return self.system_dyn.y
        else
            # TODO: raise 
            return -1

    
