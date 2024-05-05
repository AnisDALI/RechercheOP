import numpy as np

def calculate_potentials(cost_matrix, transport_solution):
    n = len(cost_matrix)      # number of sources
    m = len(cost_matrix[0])   # number of destinations
    num_vars = n + m
    A = np.zeros((n + m - 1, n + m))  # one less because we know one potential already
    b = np.zeros(n + m - 1)
    
    # Add equations for each transported cell
    eq = 0
    for i in range(n):
        for j in range(m):
            if transport_solution[i][j] > 0:
                A[eq, i] = 1
                A[eq, n + j] = 1
                b[eq] = cost_matrix[i][j]
                eq += 1
                
    # Fix potential u_0 = 0 (by not adding it into the matrix)

    # Solve the system
    potentials = np.linalg.lstsq(A[:, 1:], b, rcond=None)[0]  # solving for all but the first potential
    return np.insert(potentials, 0, 0)  # insert u_0 = 0 at the first position

# Example usage:
cost_matrix = np.array([[30, 20], [10, 50]])
transport_solution = np.array([[100, 0], [0, 100]])
potentials = calculate_potentials(cost_matrix, transport_solution)
print("Potentials:", potentials)


def calculate_reduced_costs(cost_matrix, potentials):
    n = len(cost_matrix)
    m = len(cost_matrix[0])
    reduced_costs = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            reduced_costs[i][j] = cost_matrix[i][j] - (potentials[i] + potentials[n + j])
    return reduced_costs

reduced_costs = calculate_reduced_costs(cost_matrix, potentials)
print("Reduced Costs:\n", reduced_costs)
