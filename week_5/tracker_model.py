import numpy as np
from scipy.optimize import minimize

class TrackerModel:
    def __init__(self, A_continuos, B_continuos, C_continuos, Q, R, N, q, m, n, delta):
        A_cont = A_continuos
        B_cont = B_continuos
        C_cont = C_continuos
        self.Q = Q
        self.R = R
        self.N = N # prediction horizon
        self.q = q # number of outputs  
        self.m = m # Number of control inputs
        self.orig_n = n # Number of states#
        self.delta = delta

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

        # Update obj.n to the extended state dimension
        self.n = A.shape[0]  # Extended state dimensio


    def tracker_std(self,S_bar, T_bar, Q_hat, Q_bar, R_bar):
        # Compute H
        H = R_bar + S_bar.T @ Q_bar @ S_bar
        
        # Compute F_tra
        first = -Q_hat @ S_bar
        second = T_bar.T @ Q_bar @ S_bar
        F_tra = np.vstack([first, second])  # Stack first and second vertically
        
        return H, F_tra

    #def compute_H_and_F(self, S_bar, T_bar, Q_bar, R_bar):
        # Compute H
    #    H = np.dot(S_bar.T, np.dot(Q_bar, S_bar)) + R_bar

        # Compute F
    #    F = np.dot(S_bar.T, np.dot(Q_bar, T_bar))

    #    return H, F

    def propagation_model_tracker_fixed_std(obj):
        # Determine sizes and initialize matrices
        S_bar = np.zeros((obj.n * obj.N, obj.m * obj.N))
        S_bar_C = np.zeros((obj.q * obj.N, obj.m * obj.N))
        T_bar = np.zeros((obj.n * obj.N, obj.n))
        T_bar_C = np.zeros((obj.q * obj.N, obj.n))
        Q_hat = np.zeros((obj.q * obj.N, obj.n * obj.N))
        Q_bar = np.zeros((obj.n * obj.N, obj.n * obj.N))
        R_bar = np.zeros((obj.m * obj.N, obj.m * obj.N))

        # Loop to calculate matrices
        for k in range(1, obj.N + 1):
            for j in range(1, k + 1):
                idx_row_S = slice(obj.n * (k-1), obj.n * k)
                idx_col_S = slice(obj.m * (k-j), obj.m * (k-j+1))
                S_bar[idx_row_S, idx_col_S] = np.linalg.matrix_power(obj.A, j-1) @ obj.B

                idx_row_SC = slice(obj.q * (k-1), obj.q * k)
                S_bar_C[idx_row_SC, idx_col_S] = obj.C @ np.linalg.matrix_power(obj.A, j-1) @ obj.B

            idx_row_T = slice(obj.n * (k-1), obj.n * k)
            T_bar[idx_row_T, :] = np.linalg.matrix_power(obj.A, k)

            idx_row_TC = slice(obj.q * (k-1), obj.q * k)
            T_bar_C[idx_row_TC, :] = obj.C @ np.linalg.matrix_power(obj.A, k)

            idx_row_QH = slice(obj.q * (k-1), obj.q * k)
            idx_col_QH = slice(obj.n * (k-1), obj.n * k)
            Q_hat[idx_row_QH, idx_col_QH] = obj.Q @ obj.C

            idx_row_col_QB = slice(obj.n * (k-1), obj.n * k)
            Q_bar[idx_row_col_QB, idx_row_col_QB] = obj.C.T @ obj.Q @ obj.C

            idx_row_col_R = slice(obj.m * (k-1), obj.m * k)
            R_bar[idx_row_col_R, idx_row_col_R] = obj.R

        return S_bar, S_bar_C, T_bar, T_bar_C, Q_hat, Q_bar, R_bar
    

    

    def computesolution(self,x_ref, x_cur, u_cur, H, F_tra):
        F = np.dot(np.hstack([x_ref, x_cur, u_cur]), F_tra)

        def objective(x, H, F):
            # Objective function: 0.5 * x^T * H * x + f^T * x
            return 0.5 * np.dot(x.T, np.dot(H, x)) + np.dot(F, x)

        # Initial guess (make sure it's a sensible starting point)
        x0 = np.zeros(70)  # Assuming u_cur has the correct dimension

        # Run the optimization
        result = minimize(fun=objective, x0=x0, args=(H, F), method='SLSQP')

        u_star = result.x
        return u_star
