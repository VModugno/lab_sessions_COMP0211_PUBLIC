from scipy.signal import place_poles
import numpy as np

class Observer:
    def __init__(self, A, B, C,dt,x_hat_init):
        self.A = A
        self.B = B
        self.C = C
        self.L = None
        self.x_hat = x_hat_init
        self.dt = dt

        
    def update(self, u, y):
        if self.L is None:
            raise ValueError("Observer gains have not been computed.")
        self.x_hat = self.x_hat + (self.A @ self.x_hat + self.B.flatten() * u - self.L @ (self.C @ self.x_hat - y)) * self.dt
        y_hat = self.C @ self.x_hat
        
        return self.x_hat, y_hat
    # this function assign the poles to the observer specified in lambda_1 and lambda_2
    def ComputeObserverGains(self,lambda_1,lambda_2):
        # Compute the observer gain L
        # Place the eigenvalues of (A - L C) at desired locations
        place_obj = place_poles(self.A.T, self.C.T, [lambda_1, lambda_2])
        self.L = place_obj.gain_matrix.T
        print("Observer Gain L:")
        print(self.L)

    def CheckDesiredPolynomials(self,lambda_1,lambda_2):
        # Compute the characteristic polynomial of (A - L C)
        # Desired characteristic polynomial coefficients
        desired_char_poly = np.convolve([1, -lambda_1], [1, -lambda_2])
        desired_char_eq = np.poly1d(desired_char_poly)