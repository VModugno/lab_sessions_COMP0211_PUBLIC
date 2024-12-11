import numpy as np
from scipy.optimize import minimize
    
class RegulatorModel:
    def __init__(self, N, q, m, n,constr_flag=False):
        self.A = None # state transition matrix
        self.B = None # input matrix
        self.C = None # output matrix
        self.Q = None # cost matrix
        self.R = None # cost matrix
        self.W = None # constraints matrix
        self.G = None # constraints matrix
        self.S = None # constraints matrix
        self.H = None # objective function matrix
        self.F = None
        self.F_ref = None
        self.T_bar = None
        self.S_bar = None
        self.R_bar = None
        self.Q_bar = None
        self.x_ref_bar = None  
        self.N = N
        self.q = q #  output dimension
        self.m = m #  input dimension
        self.n = n #  state dimension
        self.constr_flag = constr_flag

    def compute_H_and_F(self):
        # Compute H
        self.H = np.dot(self.S_bar.T, np.dot(self.Q_bar, self.S_bar)) + self.R_bar

        if self.x_ref_bar is not None:
           self.F_ref = np.dot(self.S_bar.T,np.dot(self.Q_bar, self.x_ref_bar))
           self.F = np.dot(self.S_bar.T, np.dot(self.Q_bar, self.T_bar))

        # Compute F transpose
        self.F = np.dot(self.S_bar.T, np.dot(self.Q_bar, self.T_bar))


    def propagation_model_regulator_fixed_std(self, x_ref=None):
        S_bar = np.zeros((self.N*self.q, self.N*self.m))
        T_bar = np.zeros((self.N*self.q, self.n))
        Q_bar = np.zeros((self.N*self.q, self.N*self.q))
        R_bar = np.zeros((self.N*self.m, self.N*self.m))

        for k in range(1, self.N + 1):
            for j in range(1, k + 1):
                S_bar[(k-1)*self.q:k*self.q, (k-j)*self.m:(k-j+1)*self.m] = np.dot(np.dot(self.C, np.linalg.matrix_power(self.A, j-1)), self.B)

            T_bar[(k-1)*self.q:k*self.q, :self.n] = np.dot(self.C, np.linalg.matrix_power(self.A, k))

            Q_bar[(k-1)*self.q:k*self.q, (k-1)*self.q:k*self.q] = self.Q
            R_bar[(k-1)*self.m:k*self.m, (k-1)*self.m:k*self.m] = self.R

        self.S_bar = S_bar
        self.T_bar = T_bar
        self.Q_bar = Q_bar
        self.R_bar = R_bar
        if x_ref is not None:
            self.x_ref_bar = np.kron(np.ones((self.N, 1)), x_ref)
            self.x_ref_bar = self.x_ref_bar.flatten().reshape(-1, 1)
    
    def setSystemMatrices(self,delta_t,A,B):
        """
        Get the system matrices A and B according to the dimensions of the state and control input.
        
        Parameters:
        num_states, number of system states
        num_controls, number oc conttrol inputs
        cur_x, current state around which to linearize
        cur_u, current control input around which to linearize
       
        
        Returns:
        A: State transition matrix
        B: Control input matrix
        """
       
        num_states = self.n
        num_controls = self.m
        num_joints = num_controls
        num_outputs = self.q
        
        # Initialize A matrix (discrete-time)
        A_disc = np.eye(num_states) + delta_t * A
        
        # Initialize B matrix (discrete-time)
        B_disc = B * delta_t
        
        self.A = A_disc
        self.B = B_disc
        # here we assume i have a full state feedback (and if i do not we can use the observer to estimate the states)
        self.C = np.eye(num_outputs)
    
    def setCostMatrices(self, Qcoeff, Rcoeff):
        """
        Set the cost matrices Q and R for the MPC controller.

        Parameters:
        Qcoeff: float or array-like
            State cost coefficient(s). If scalar, the same weight is applied to all states.
            If array-like, should have a length equal to the number of states.

        Rcoeff: float or array-like
            Control input cost coefficient(s). If scalar, the same weight is applied to all control inputs.
            If array-like, should have a length equal to the number of control inputs.

        Sets:
        self.Q: ndarray
            State cost matrix.
        self.R: ndarray
            Control input cost matrix.
        """
        
        num_states = self.n
        num_controls = self.m

        # Process Qcoeff
        if np.isscalar(Qcoeff):
            # If Qcoeff is a scalar, create an identity matrix scaled by Qcoeff
            Q = Qcoeff * np.eye(num_states)
        else:
            # Convert Qcoeff to a numpy array
            Qcoeff = np.array(Qcoeff)
            if Qcoeff.ndim != 1 or len(Qcoeff) != num_states:
                raise ValueError(f"Qcoeff must be a scalar or a 1D array of length {num_states}")
            # Create a diagonal matrix with Qcoeff as the diagonal elements
            Q = np.diag(Qcoeff)

        # Process Rcoeff
        if np.isscalar(Rcoeff):
            # If Rcoeff is a scalar, create an identity matrix scaled by Rcoeff
            R = Rcoeff * np.eye(num_controls)
        else:
            # Convert Rcoeff to a numpy array
            Rcoeff = np.array(Rcoeff)
            if Rcoeff.ndim != 1 or len(Rcoeff) != num_controls:
                raise ValueError(f"Rcoeff must be a scalar or a 1D array of length {num_controls}")
            # Create a diagonal matrix with Rcoeff as the diagonal elements
            R = np.diag(Rcoeff)

        # Assign the matrices to the object's attributes
        self.Q = Q
        self.R = R


    def regulator_G_std(self, S_bar):
        G = np.vstack([S_bar, -S_bar, np.eye(self.N * self.m), -np.eye(self.N * self.m)])
        return G

    def regulator_S_std(self, T_bar):
        S = np.vstack([
            -T_bar,
            T_bar,
            np.zeros((self.N * self.m, self.n)),
            np.zeros((self.N * self.m, self.n))
        ])
        return S

    
    # B_In input bound constraints (dict): A dictionary containing the input bound constraints.
    # B_Out output bound constraints (dict): A dictionary containing the output bound constraints.
    def regulator_W_std(self, B_Out, B_In):
        # Check if 'min' fields are not empty and exist in B_Out and B_In
        out_min_present = 'min' in B_Out and B_Out['min'] is not None
        in_min_present = 'min' in B_In and B_In['min'] is not None

        # if out_min_present:
        #     if in_min_present:  # out min true, in min true
        #         block1 = np.kron(np.ones((self.N, 1)), B_Out['max'])
        #         block2 = np.kron(np.ones((self.N, 1)), -B_Out['min'])
        #         block3 = np.kron(np.ones((self.N, 1)), B_In['max'])
        #         block4 = np.kron(np.ones((self.N, 1)), -B_In['min'])
        #     else:  # out min true, in min false
        #         W = np.vstack([
        #             np.kron(np.ones((self.N, 1)), B_Out['max']),
        #             np.kron(np.ones((self.N, 1)), -B_Out['min']),
        #             np.kron(np.ones((self.N, 1)), B_In['max']),
        #             np.kron(np.ones((self.N, 1)), B_In['max'])
        #         ])
        # elif in_min_present:  # out min false, in min true
        #     W = np.vstack([
        #         np.kron(np.ones((self.N, 1)), B_Out['max']),
        #         np.kron(np.ones((self.N, 1)), B_Out['max']),
        #         np.kron(np.ones((self.N, 1)), B_In['max']),
        #         np.kron(np.ones((self.N, 1)), B_In['min'])
        #     ])
        # else:  # out min false, in min false
        #     W = np.vstack([
        #         np.kron(np.ones((self.N, 1)), B_Out['max']),
        #         np.kron(np.ones((self.N, 1)), B_Out['max']),
        #         np.kron(np.ones((self.N, 1)), B_In['max']),
        #         np.kron(np.ones((self.N, 1)), B_In['max'])
        #     ])

        #if out_min_present:
            #if in_min_present:  # out min true, in min true
        block1 = np.kron(np.ones((self.N, 1)), B_Out['max'])
        block2 = np.kron(np.ones((self.N, 1)), -B_Out['min'])
        block3 = np.kron(np.ones((self.N, 1)), B_In['max'])
        block4 = np.kron(np.ones((self.N, 1)), -B_In['min'])
        #     else:  # out min true, in min false
        #         block1 = np.kron(np.ones((self.N, 1)), B_Out['max'])
        #         block2 = np.kron(np.ones((self.N, 1)), -B_Out['min'])
        #         block3 = np.kron(np.ones((self.N, 1)), B_In['max'])
        #         block4 = np.kron(np.ones((self.N, 1)), B_In['max'])  # Using max since min is not present
        # else:
        #     if in_min_present:  # out min false, in min true
        #         block1 = np.kron(np.ones((self.N, 1)), B_Out['max'])
        #         block2 = np.kron(np.ones((self.N, 1)), B_Out['max'])  # Repeating max since min is not present
        #         block3 = np.kron(np.ones((self.N, 1)), B_In['max'])
        #         block4 = np.kron(np.ones((self.N, 1)), -B_In['min'])
        #     else:  # out min false, in min false
        #         block1 = np.kron(np.ones((self.N, 1)), B_Out['max'])
        #         block2 = np.kron(np.ones((self.N, 1)), B_Out['max'])
        #         block3 = np.kron(np.ones((self.N, 1)), B_In['max'])
        #         block4 = np.kron(np.ones((self.N, 1)), B_In['max'])

        vec1 = block1.flatten().reshape(-1, 1)  
        vec2 = block2.flatten().reshape(-1, 1)
        vec3 = block3.flatten().reshape(-1, 1)
        vec4 = block4.flatten().reshape(-1, 1)

        # Step 3: Vertically stack all vectors
        W = np.vstack([vec1, vec2, vec3, vec4])  
        return W

    # added function to compute constraints matrices
    def setConstraintsMatrices(self,B_in,B_out):

        self.G = self.regulator_G_std(self.S_bar)
        self.S = self.regulator_S_std(self.T_bar)
        self.W = self.regulator_W_std(B_out, B_in)
    

    # add constraints to the optimization problem
    def compute_solution(self, x0_mpc):
         # Update the objective function based on the given equation
         def objective(z, x0_mpc,H,F, F_ref=None):
             if F_ref is not None:
                 F_full = -F_ref + np.dot(F,x0_mpc)[:, np.newaxis]
                 return 0.5 * np.dot(z.T, np.dot(H, z)) +  np.dot(F_full.T, z)
             else:  
                 return 0.5 * np.dot(z.T, np.dot(H, z)) + np.dot(x0_mpc.T, np.dot(F.T, z))

         if self.constr_flag:
             # Constraint function
             def constraint(z, G, W, S, x0_mpc):
                 W_flat = W.flatten()
                 # here we use inequality constraints of type Gz <= W + Sx0      +W+Sx0 -Gz >=  0
                 return (+W_flat + S @ x0_mpc) -G @ z

             # Constraints dictionary
             cons = {'type': 'ineq', 'fun': constraint, 'args': (self.G, self.W, self.S, x0_mpc)}
         else:
             cons = None  # No constraints

         # Initial guess (size should be determined by problem dimension)
         z0 = np.zeros(self.m * self.N)  # Assuming z has dimensions m * N

         # Options to increase numerical accuracy
         options_dict = {
             'ftol': 1e-12,        # Increase precision goal for the objective function 
             'eps': 1e-12,         # Smaller step size for gradient approximation
             'maxiter': 1000,      # Allow more iterations to find a more accurate solution
             'disp': True          # Display convergence messages for debugging
         }

         # Run the optimization
         result = minimize(fun=objective, x0=z0, args=(x0_mpc,self.H,self.F, self.F_ref), method='SLSQP', constraints=cons,options=options_dict)

         z_star = result.x
         return z_star


    # def compute_solution(self, x0_mpc):
    #     # Precompute the constant term: (T_bar x0 - x_ref_bar)^T Q_bar (T_bar x0 - x_ref_bar)
    #     # This makes sure the cost reflects the proper nonnegative quadratic form.
    #     diff = self.T_bar @ x0_mpc
    #     diff = diff[:, np.newaxis]
    #     diff = diff - (self.x_ref_bar if self.x_ref_bar is not None else 0)
    #     #diff = self.T_bar @ x0_mpc - (self.x_ref_bar if self.x_ref_bar is not None else 0)
    #     self.constant_term = diff.T @ self.Q_bar @ diff

    #     def objective(z, x0_mpc, H, F, F_ref=None, const_term=0):
    #         # Quadratic form
    #         quad = 0.5 * z.T @ (H @ z)
    #         # Linear terms
    #         if F_ref is not None:
    #             # (F x_0 - F_ref)^T u
    #             linear = ( (F @ x0_mpc)[:, np.newaxis] - F_ref ).T @ z
    #         else:
    #             # If no ref, linear = (F x_0)^T u
    #             linear = (F @ x0_mpc).T @ z

    #         return quad + linear + const_term

    #     if self.constr_flag:
    #         def constraint(z, G, W, S, x0_mpc):
    #             W_flat = W.flatten()
    #             return (W_flat + S @ x0_mpc) - G @ z
    #         cons = {'type': 'ineq', 'fun': constraint, 'args': (self.G, self.W, self.S, x0_mpc)}
    #     else:
    #         cons = None

    #     z0 = np.zeros(self.m * self.N)
    #     options_dict = {
    #         'ftol': 1e-12,
    #         'eps': 1e-12,
    #         'maxiter': 1000,
    #         'disp': True
    #     }

    #     result = minimize(
    #         fun=objective,
    #         x0=z0,
    #         args=(x0_mpc, self.H, self.F, self.F_ref, self.constant_term),
    #         method='SLSQP',
    #         constraints=cons,
    #         options=options_dict
    #     )

    #     z_star = result.x
    #     return z_star
    
    def getCostMatrices(self):
        return self.Q,self.R
    
    def getDiscreteA(self):
        return self.A
    
    def getDiscreteB(self):
        return self.B
    
    def checkStability(self):
        eigvals = np.linalg.eigvals(self.A)
        if np.all(np.abs(eigvals) < 1):
            print("The system is stable.")
        else:
            print("The system is unstable.")

    def checkControllabilityDiscrete(self):
        C = np.hstack([self.B,self.A @ self.B])
        if np.linalg.matrix_rank(C) == 2:
            print("The discrete system is controllable.")
        else:
            print("The discrete system is not controllable.")