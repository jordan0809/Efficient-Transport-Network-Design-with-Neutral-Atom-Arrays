import pulser
from pulser import Pulse, Sequence, Register
import numpy as np

# Define the number of stations
N = 10

# Create a dictionary of qubit IDs with position coordinates
qubit_dict = {f'q{i}': (np.random.rand(), np.random.rand()) for i in range(N)}

# Create a register with binary variables Xi for each qubit
qubit_register = Register(qubit_dict)

# Create a pulse sequence and specify the device (Ensure device parameters are valid)
device_params = pulser.devices.Chadoq2
seq = Sequence(qubit_register, device_params)

# Define the binary variables Xi for each qubit
Xi = [qubit_register[qubit_id] for qubit_id in qubit_dict.keys()]

# Define the weights (distances) between stations
distances = np.random.randint(1, 100, size=(N, N))
np.fill_diagonal(distances, 0)  # Ensure zero distance for self-loops

# Define the cost Hamiltonian based on the objective function
H = 0
for i in range(N):
    for j in range(i + 1, N):
        H += ((1 - Xi[i]) * (1 - Xi[j]) + Xi[i] * Xi[j]) * distances[i][j]

# Add the constraint for the sum of Xi to be N/2
constraint = sum(Xi) == N // 2

# Add the cost Hamiltonian and constraint to the sequence
seq.add(H, 'rydberg')
seq.add(constraint, 'rydberg')

# Optimize the cost Hamiltonian to minimize the sum of overall distances
result = seq.optimize_pulses()

# Access the optimized binary variables Xi
optimized_Xi = result.optimized_values(Xi)

# Print the random network and the optimized Xi values
print("Random Distances:")
print(distances)
print("Optimized Xi:", optimized_Xi)
