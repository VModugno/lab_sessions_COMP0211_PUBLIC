import numpy as np

class SysDyn:
    def __init__(self, J, b, K_t, K_e, R_a, L_a,d_t, x_init):
       
        self.dt = d_t
        self.A = np.array([
            [-b / J,        K_t / J],
            [-K_e / L_a,   -R_a / L_a]
        ])
        self.B = np.array([
            [0],
            [1 / L_a]
        ])
        self.C = np.array([[K_e, R_a]])
        self.x = x_init
        # observabiity
        self.O = np.vstack([
            self.C,
            self.C @ self.A
        ])
        # checking observability
        if np.linalg.matrix_rank(self.O) != 2:
            print("The system is not observable with the given C matrix.")
        else:
            print("The system is observable.")

    def step(self, u):
        dx_dt = self.A @ self.x + self.B.flatten() * u
        # Include load torque if necessary (here T_L is assumed zero)
        # dx_dt[0] -= T_L / J
        
        # Update system state using Euler integration
        self.x = self.x + self.dt * dx_dt
        output = self.C @ self.x
        return output
    
    def getCurrentState(self):
        return self.x
    
    def getA(self):
        return self.A
    
    def getB(self):
        return self.B
    
    def checkControlabilityContinuos(self):
        C = np.hstack([self.B,self.A @ self.B])
        if np.linalg.matrix_rank(C) != 2:
            print("The continous system is not controllable with the given B matrix.")
        else:
            print("The continous system is controllable.")
    