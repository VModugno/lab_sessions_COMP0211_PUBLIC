import numpy as np
    
class RegulatorModel:
    def __init__(self, N, q, m, n):
        self.A = None
        self.B = None
        self.C = None
        self.Q = None
        self.R = None
        self.N = N
        self.q = q #  output dimension
        self.m = m #  input dimension
        self.n = n #  state dimension

    def compute_H_and_F(self, S_bar, T_bar, Q_bar, R_bar):
        # Compute H
        H = np.dot(S_bar.T, np.dot(Q_bar, S_bar)) + R_bar

        # Compute F
        F = np.dot(S_bar.T, np.dot(Q_bar, T_bar))

        return H, F

    def propagation_model_regulator_fixed_std(self):
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

        return S_bar, T_bar, Q_bar, R_bar
    
    def updateSystemMatrices(self,sim,cur_x,cur_u):
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
        # Check if state_x_for_linearization and cur_u_for_linearization are provided
        if cur_x is None or cur_u is None:
            raise ValueError(
                "state_x_for_linearization and cur_u_for_linearization are not specified.\n"
                "Please provide the current state and control input for linearization.\n"
                "Hint: Use the goal state (e.g., zeros) and zero control input at the beginning.\n"
                "Also, ensure that you implement the linearization logic in the updateSystemMatrices function."
            )

        A =[]
        B = []
        num_states = self.n
        num_controls = self.m
        num_outputs = self.q
        delta_t = sim.GetTimeStep()
        v0 = cur_x[0]
        theta0 = cur_x[2]
        # get A and B matrices by linearinzing the cotinuous system dynamics
        # The linearized continuous-time system is:

        # \[
        # \dot{\mathbf{x}} = A_c (\mathbf{x} - \mathbf{x}_0) + B_c (\mathbf{u} - \mathbf{u}_0).
        # \]

        # \textbf{Compute \( A_c = \left. \dfrac{\partial \mathbf{f}}{\partial \mathbf{x}} \right|_{(\mathbf{x}_0, \mathbf{u}_0)} \):}

        # \[
        # A_c = \begin{bmatrix}
        # \frac{\partial \dot{x}}{\partial x} & \frac{\partial \dot{x}}{\partial y} & \frac{\partial \dot{x}}{\partial \theta} \\
        # \frac{\partial \dot{y}}{\partial x} & \frac{\partial \dot{y}}{\partial y} & \frac{\partial \dot{y}}{\partial \theta} \\
        # \frac{\partial \dot{\theta}}{\partial x} & \frac{\partial \dot{\theta}}{\partial y} & \frac{\partial \dot{\theta}}{\partial \theta}
        # \end{bmatrix}.
        # \]

        # Compute the partial derivatives:

        # \begin{align*}
        # \frac{\partial \dot{x}}{\partial x} &= 0, & \frac{\partial \dot{x}}{\partial y} &= 0, & \frac{\partial \dot{x}}{\partial \theta} &= -v_0 \sin(\theta_0), \\
        # \frac{\partial \dot{y}}{\partial x} &= 0, & \frac{\partial \dot{y}}{\partial y} &= 0, & \frac{\partial \dot{y}}{\partial \theta} &= v_0 \cos(\theta_0), \\
        # \frac{\partial \dot{\theta}}{\partial x} &= 0, & \frac{\partial \dot{\theta}}{\partial y} &= 0, & \frac{\partial \dot{\theta}}{\partial \theta} &= 0.
        # \end{align*}

        # Thus,

        # \[
        # A_c = \begin{bmatrix}
        # 0 & 0 & -v_0 \sin(\theta_0) \\
        # 0 & 0 & v_0 \cos(\theta_0) \\
        # 0 & 0 & 0
        # \end{bmatrix}.
        # \]

        # \textbf{Compute \( B_c = \left. \dfrac{\partial \mathbf{f}}{\partial \mathbf{u}} \right|_{(\mathbf{x}_0, \mathbf{u}_0)} \):}

        # \[
        # B_c = \begin{bmatrix}
        # \frac{\partial \dot{x}}{\partial v} & \frac{\partial \dot{x}}{\partial \omega} \\
        # \frac{\partial \dot{y}}{\partial v} & \frac{\partial \dot{y}}{\partial \omega} \\
        # \frac{\partial \dot{\theta}}{\partial v} & \frac{\partial \dot{\theta}}{\partial \omega}
        # \end{bmatrix}.
        # \]

        # Compute the partial derivatives:

        # \begin{align*}
        # \frac{\partial \dot{x}}{\partial v} &= \cos(\theta_0), & \frac{\partial \dot{x}}{\partial \omega} &= 0, \\
        # \frac{\partial \dot{y}}{\partial v} &= \sin(\theta_0), & \frac{\partial \dot{y}}{\partial \omega} &= 0, \\
        # \frac{\partial \dot{\theta}}{\partial v} &= 0, & \frac{\partial \dot{\theta}}{\partial \omega} &= 1.
        # \end{align*}

        # Thus,

        # \[
        # B_c = \begin{bmatrix}
        # \cos(\theta_0) & 0 \\
        # \sin(\theta_0) & 0 \\
        # 0 & 1
        # \end{bmatrix}.
        # \]



        # then linearize A and B matrices
        #\[
        # A = I + \Delta t \cdot A_c,
        # \]
        # \[
        # B = \Delta t \cdot B_c,
        # \]

        # where \( I \) is the identity matrix.

        # Compute \( A \):

        # \[
        # A = \begin{bmatrix}
        # 1 & 0 & -v_0 \Delta t \sin(\theta_0) \\
        # 0 & 1 & v_0 \Delta t \cos(\theta_0) \\
        # 0 & 0 & 1
        # \end{bmatrix}.
        # \]

        # Compute \( B \):

        # \[
        # B = \begin{bmatrix}
        # \Delta t \cos(\theta_0) & 0 \\
        # \Delta t \sin(\theta_0) & 0 \\
        # 0 & \Delta t
        # \end{bmatrix}.
        # \]
        
        #updating the state and control input matrices
       


        self.A = A
        self.B = B
        self.C = np.eye(num_outputs)
        



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
        import numpy as np

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