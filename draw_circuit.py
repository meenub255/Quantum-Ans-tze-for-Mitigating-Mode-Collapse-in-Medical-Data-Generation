from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
import matplotlib.pyplot as plt

# 1. Initialize a 2-qubit quantum circuit
qc = QuantumCircuit(2)

# 2. Stage 1: Feature Mapping / State Preparation
# Define Qiskit Parameters for the classical inputs to render text labels
x0 = Parameter('Input_x0')
x1 = Parameter('Input_x1')

qc.rx(x0, 0)
qc.rx(x1, 1)
qc.barrier() # Visual stage divider

# 3. Stage 2: Variational Layers & Quantum Entanglement
w0 = Parameter('w_0')
w1 = Parameter('w_1')

qc.rx(w0, 0)
qc.ry(w1, 1)
qc.cx(0, 1) # CNOT gate (Control: q0, Target: q1)
qc.barrier()

# Deepening layer parameters
w2 = Parameter('w_2')
w3 = Parameter('w_3')

qc.ry(w2, 0)
qc.rx(w3, 1)
qc.cx(1, 0) # CNOT gate (Control: q1, Target: q0)
qc.barrier()

# 4. Stage 3: Measurement mapping to Pauli-Z expectations
qc.measure_all()

# 5. Draw and save using matplotlib engine with 'iqp' color palette layout
qc.draw(output='mpl', style='iqp')

# Save vector graphic output
plt.savefig('quantum_gan_circuit.png', bbox_inches='tight', dpi=300)
print("✅ Beautiful graphical circuit schematic saved as 'quantum_gan_circuit.png'!")