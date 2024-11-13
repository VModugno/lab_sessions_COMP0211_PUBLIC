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

    if out_min_present:
        if in_min_present:  # out min true, in min true
            W = np.vstack([
                np.kron(np.ones((self.N, 1)), B_Out['max']),
                np.kron(np.ones((self.N, 1)), -B_Out['min']),
                np.kron(np.ones((self.N, 1)), B_In['max']),
                np.kron(np.ones((self.N, 1)), -B_In['min'])
            ])
        else:  # out min true, in min false
            W = np.vstack([
                np.kron(np.ones((self.N, 1)), B_Out['max']),
                np.kron(np.ones((self.N, 1)), -B_Out['min']),
                np.kron(np.ones((self.N, 1)), B_In['max']),
                np.kron(np.ones((self.N, 1)), B_In['max'])
            ])
    elif in_min_present:  # out min false, in min true
        W = np.vstack([
            np.kron(np.ones((self.N, 1)), B_Out['max']),
            np.kron(np.ones((self.N, 1)), B_Out['max']),
            np.kron(np.ones((self.N, 1)), B_In['max']),
            np.kron(np.ones((self.N, 1)), B_In['min'])
        ])
    else:  # out min false, in min false
        W = np.vstack([
            np.kron(np.ones((self.N, 1)), B_Out['max']),
            np.kron(np.ones((self.N, 1)), B_Out['max']),
            np.kron(np.ones((self.N, 1)), B_In['max']),
            np.kron(np.ones((self.N, 1)), B_In['max'])
        ])
    return W