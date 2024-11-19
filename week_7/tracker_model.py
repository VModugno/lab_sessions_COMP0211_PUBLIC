import numpy as np
from scipy.optimize import minimize

class TrackerModel:
    def __init__(self, N, q, m, n, delta,constr_flag=True):
        #A_cont = A_continuos
        #B_cont = B_continuos
        #C_cont = C_continuos
        self.A = None # state transition matrix
        self.B = None # input matrix
        self.C = None # output matrix
        self.Q = None # cost matrix
        self.R = None # cost matrix
        self.W = None # constraints matrix
        self.G = None # constraints matrix
        self.S = None # constraints matrix  
        self.N = N # prediction horizon
        self.q = q # number of outputs  
        self.m = m # Number of control inputs
        self.orig_n = n # Number of states#
        self.delta = delta
        self.constr_flag = constr_flag

       

    def tracker_std(self,S_bar, T_bar, Q_hat, Q_bar, R_bar):
        # Compute H
        H = R_bar + S_bar.T @ Q_bar @ S_bar
        
        # Compute F_tra
        first = -Q_hat @ S_bar
        second = T_bar.T @ Q_bar @ S_bar
        F_tra = np.vstack([first, second])  # Stack first and second vertically
        
        return H, F_tra
    
    def setSystemMatrices(self,damping_coefficients=None):

        num_states = 2 * self.m
        num_controls = self.m
        num_joints = self.m
        
        
        # Initialize A matrix
        A_cont = np.zeros((num_states,num_states))
        
        # Upper right quadrant of A (position affected by velocity)
        A_cont[:num_joints, num_joints:] = np.eye(num_joints) 
        
        # Lower right quadrant of A (velocity affected by damping)
        #if damping_coefficients is not None:
        #    damping_matrix = np.diag(damping_coefficients)
        #    A[num_joints:, num_joints:] = np.eye(num_joints) - time_step * damping_matrix
        
        # Initialize B matrix
        B_cont = np.zeros((num_states, num_controls))
        
        # Lower half of B (control input affects velocity)
        B_cont[num_joints:, :] = np.eye(num_controls) 

        C_cont = np.eye(num_states)

         # Compute matrix A
        A_upper_left = np.eye(self.orig_n) + self.delta * A_cont
        A_upper_right = self.delta * B_cont
        A_lower_left = np.zeros((self.m, self.orig_n))
        A_lower_right = np.eye(self.m)

        # Combine blocks to form A
        A_upper = np.hstack((A_upper_left, A_upper_right))
        A_lower = np.hstack((A_lower_left, A_lower_right))
        A = np.vstack((A_upper, A_lower))

        # Alternatively, using np.block for clarity
        self.A = np.block([
            [np.eye(self.orig_n) + self.delta * A_cont, self.delta * B_cont],
            [np.zeros((self.m, self.orig_n)), np.eye(self.m)]
        ])

        # Compute matrix B
        self.B = np.vstack((self.delta * B_cont, np.eye(self.m)))

        # Compute matrix C
        self.C = np.hstack((C_cont, np.zeros((self.q, self.m))))

        # Update self.n to the extended state dimension 
        self.n = A.shape[0]  # Extended state dimension

    # TODO you can change this function to allow for more passing a vector of gains
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

        num_states = self.orig_n
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

    def propagation_model_tracker_fixed_std(self):
        # Determine sizes and initialize matrices
        S_bar = np.zeros((self.n * self.N, self.m * self.N))
        S_bar_C = np.zeros((self.q * self.N, self.m * self.N))
        T_bar = np.zeros((self.n * self.N, self.n))
        T_bar_C = np.zeros((self.q * self.N, self.n))
        Q_hat = np.zeros((self.q * self.N, self.n * self.N))
        Q_bar = np.zeros((self.n * self.N, self.n * self.N))
        R_bar = np.zeros((self.m * self.N, self.m * self.N))

        # Loop to calculate matrices
        for k in range(1, self.N + 1):
            for j in range(1, k + 1):
                idx_row_S = slice(self.n * (k-1), self.n * k)
                idx_col_S = slice(self.m * (k-j), self.m * (k-j+1))
                S_bar[idx_row_S, idx_col_S] = np.linalg.matrix_power(self.A, j-1) @ self.B

                idx_row_SC = slice(self.q * (k-1), self.q * k)
                S_bar_C[idx_row_SC, idx_col_S] = self.C @ np.linalg.matrix_power(self.A, j-1) @ self.B

            idx_row_T = slice(self.n * (k-1), self.n * k)
            T_bar[idx_row_T, :] = np.linalg.matrix_power(self.A, k)

            idx_row_TC = slice(self.q * (k-1), self.q * k)
            T_bar_C[idx_row_TC, :] = self.C @ np.linalg.matrix_power(self.A, k)

            idx_row_QH = slice(self.q * (k-1), self.q * k)
            idx_col_QH = slice(self.n * (k-1), self.n * k)
            Q_hat[idx_row_QH, idx_col_QH] = self.Q @ self.C

            idx_row_col_QB = slice(self.n * (k-1), self.n * k)
            Q_bar[idx_row_col_QB, idx_row_col_QB] = self.C.T @ self.Q @ self.C

            idx_row_col_R = slice(self.m * (k-1), self.m * k)
            R_bar[idx_row_col_R, idx_row_col_R] = self.R

        return S_bar, S_bar_C, T_bar, T_bar_C, Q_hat, Q_bar, R_bar
    
    def tracker_G_std(self, S_bar_C):
        #G = np.vstack([S_bar_C, -S_bar_C, np.eye(self.N * self.m), -np.eye(self.N * self.m)])
        G = np.vstack([S_bar_C,-S_bar_C,np.tril(np.ones((self.N * self.m, self.N * self.m))),-np.tril(np.ones((self.N * self.m, self.N * self.m)))])
        return G

    def tracker_S_std(self, T_bar_C):
        S = np.vstack([
            -T_bar_C,
            T_bar_C,
            np.zeros((self.N * self.m, self.n)),
            np.zeros((self.N * self.m, self.n))
        ])
        return S

    
    # B_In input bound constraints (dict): A dictionary containing the input bound constraints.
    # B_Out output bound constraints (dict): A dictionary containing the output bound constraints.
    def tracker_W_std(self, B_Out, B_In):
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
    def setConstraintsMatrices(self,B_in,B_out,S_bar_C,T_bar_C):

        self.G = self.tracker_G_std(S_bar_C)
        self.S = self.tracker_S_std(T_bar_C)
        self.W = self.tracker_W_std(B_out, B_in)
    

    def computesolution(self,x_ref, x_cur, u_cur, H, F_tra,initial_guess=None):
        # Define the objective function
        x0_mpc_T=np.hstack([x_ref, x_cur, u_cur])
        # there is a problem here the x0_mpc should include also the reference trajectory
        # this is a problem that im carrying from 2019
        x0_mpc = np.hstack([x_cur, u_cur])
        F = np.dot(x0_mpc_T, F_tra)

        def objective(x, H, F):
            # Objective function: 0.5 * x^T * H * x + f^T * x
            return 0.5 * np.dot(x.T, np.dot(H, x)) + np.dot(F, x)

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
        if initial_guess is not None:
            z0 = initial_guess
        else:
            z0 = np.zeros(self.m * self.N)  # Assuming z has dimensions m * N

        # Options to increase numerical accuracy
        options_dict = {
            'ftol': 1e-12,        # Increase precision goal for the objective function 
            'eps': 1e-12,         # Smaller step size for gradient approximation
            'maxiter': 1000,      # Allow more iterations to find a more accurate solution
            'disp': True          # Display convergence messages for debugging
        }

        # Run the optimization
        result = minimize(fun=objective, x0=z0, args=(H, F), method='SLSQP', constraints=cons, options=options_dict)

        u_star = result.x
        return u_star
