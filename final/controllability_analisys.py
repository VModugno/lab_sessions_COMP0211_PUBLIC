import numpy as np

# Example values
v0 = 1.0             # Initial velocity
theta0 = np.pi / 4   # Initial orientation (45 degrees)
delta_t = 0.1        # Time step

# System matrices
A = np.array([
    [1, 0, -v0 * delta_t * np.sin(theta0)],
    [0, 1, v0 * delta_t * np.cos(theta0)],
    [0, 0, 1]
])

B = np.array([
    [delta_t * np.cos(theta0), 0],
    [delta_t * np.sin(theta0), 0],
    [0, delta_t]
])

# Number of states
n = A.shape[0]

# Construct the controllability matrix
controllability_matrix = B
for i in range(1, n):
    AB = np.linalg.matrix_power(A, i).dot(B)
    controllability_matrix = np.hstack((controllability_matrix, AB))

# Compute the rank
rank_of_controllability = np.linalg.matrix_rank(controllability_matrix)

print("Controllability Matrix:")
print(controllability_matrix)
print("\nRank of Controllability Matrix:", rank_of_controllability)

if rank_of_controllability == n:
    print("The system is controllable.")
else:
    print("The system is not controllable.")
