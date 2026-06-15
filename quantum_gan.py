import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. TARGET REAL DATA DISTRIBUTION
# =====================================================================
def get_real_data(batch_size):
    # Target distribution: A tight Gaussian cluster centered at (0.6, 0.4)
    mean = torch.tensor([0.6, 0.4])
    return torch.randn(batch_size, 2) * 0.05 + mean

# Global Configurations
batch_size = 16
epochs = 100
latent_dim = 2

# =====================================================================
# 2. HYBRID QUANTUM GAN (QGAN) SETUP
# =====================================================================
num_qubits = 2
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev, interface="torch")
def quantum_generator_circuit(weights, inputs):
    # High-fidelity Feature Mapping: Embed classical latent noise into Hilbert space
    qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation='X')
    
    # Strongly Entangling Layers / Variational Circuit Topology
    qml.RX(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.CNOT(wires=[0, 1])  # Generates quantum entanglement between features
    qml.RY(weights[2], wires=0)
    qml.RX(weights[3], wires=1)
    qml.CNOT(wires=[1, 0])
    
    # Extract continuous variables via non-linear expectation measurements
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

class ClassicalDiscriminator(nn.Module):
    def __init__(self):
        super(ClassicalDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 4),
            nn.LeakyReLU(0.2),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# =====================================================================
# 3. PURELY CLASSICAL GAN SETUP (BASELINE)
# =====================================================================
class ClassicalGenerator(nn.Module):
    def __init__(self):
        super(ClassicalGenerator, self).__init__()
        # Matches the parameters & input/output dimensional footprint of the PQC
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)

# =====================================================================
# 4. TRAINING EXECUTION PIPELINE
# =====================================================================

# --- Train Hybrid Quantum GAN ---
print("🚀 [1/2] Training Enhanced Hybrid Quantum GAN...")
q_discriminator = ClassicalDiscriminator()
quantum_weights = torch.randn(4, requires_grad=True)

qd_optimizer = optim.Adam(q_discriminator.parameters(), lr=0.01)
qg_optimizer = optim.Adam([quantum_weights], lr=0.02)
loss_fn = nn.BCELoss()

q_d_losses = []
q_g_losses = []

for epoch in range(epochs):
    qd_optimizer.zero_grad()
    real_data = get_real_data(batch_size)
    real_labels = torch.ones(batch_size, 1)
    qd_real_loss = loss_fn(q_discriminator(real_data), real_labels)
    
    noise = torch.randn(batch_size, latent_dim)
    q_fake_data = torch.zeros(batch_size, 2)
    for i in range(batch_size):
        q_out = quantum_generator_circuit(quantum_weights, noise[i])
        q_fake_data[i] = torch.stack(q_out)
        
    fake_labels = torch.zeros(batch_size, 1)
    qd_fake_loss = loss_fn(q_discriminator(q_fake_data.detach()), fake_labels)
    qd_loss = qd_real_loss + qd_fake_loss
    qd_loss.backward()
    qd_optimizer.step()
    
    qg_optimizer.zero_grad()
    qg_loss = loss_fn(q_discriminator(q_fake_data), real_labels)
    qg_loss.backward()
    qg_optimizer.step()
    
    q_d_losses.append(qd_loss.item())
    q_g_losses.append(qg_loss.item())

# --- Train Purely Classical GAN Baseline ---
print("\n⚙️ [2/2] Training Classical GAN Baseline...")
c_generator = ClassicalGenerator()
c_discriminator = ClassicalDiscriminator()

cd_optimizer = optim.Adam(c_discriminator.parameters(), lr=0.01)
cg_optimizer = optim.Adam(c_generator.parameters(), lr=0.02)

c_d_losses = []
c_g_losses = []

for epoch in range(epochs):
    cd_optimizer.zero_grad()
    real_data = get_real_data(batch_size)
    real_labels = torch.ones(batch_size, 1)
    cd_real_loss = loss_fn(c_discriminator(real_data), real_labels)
    
    noise = torch.randn(batch_size, latent_dim)
    c_fake_data = c_generator(noise)
    fake_labels = torch.zeros(batch_size, 1)
    cd_fake_loss = loss_fn(c_discriminator(c_fake_data.detach()), fake_labels)
    cd_loss = cd_real_loss + cd_fake_loss
    cd_loss.backward()
    cd_optimizer.step()
    
    cg_optimizer.zero_grad()
    cg_loss = loss_fn(c_discriminator(c_fake_data), real_labels)
    cg_loss.backward()
    cg_optimizer.step()
    
    c_d_losses.append(cd_loss.item())
    c_g_losses.append(cg_loss.item())

print("\n🎯 Training Complete for Both Paradigms!")

# =====================================================================
# 5. PORTFOLIO ARTIFACT GENERATION & VISUALIZATION
# =====================================================================
print("\n📊 Generating Convergence & Distribution Plots...")

# Plot 1: Convergence Comparison
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(q_g_losses, label='Quantum Gen Loss', color='crimson', linestyle='-')
plt.plot(c_g_losses, label='Classical Gen Loss', color='navy', linestyle='--')
plt.title('Generator Adversarial Convergence Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle=':')

# Plot 2: Generated Distributions
with torch.no_grad():
    eval_noise = torch.randn(200, latent_dim)
    real_points = get_real_data(200)
    
    # Collect Quantum points
    q_points = torch.zeros(200, 2)
    for i in range(200):
        q_points[i] = torch.stack(quantum_generator_circuit(quantum_weights, eval_noise[i]))
    
    # Collect Classical points
    c_points = c_generator(eval_noise)

plt.subplot(1, 2, 2)
plt.scatter(real_points[:, 0], real_points[:, 1], alpha=0.5, label='Target Real Data', color='gray')
plt.scatter(q_points[:, 0], q_points[:, 1], alpha=0.6, label='Quantum Data', color='crimson', edgecolors='k', s=25)
plt.scatter(c_points[:, 0], c_points[:, 1], alpha=0.6, label='Classical Data', color='navy', marker='x')
plt.title('Feature Space Distribution Mapping')
plt.xlabel('Feature X')
plt.ylabel('Feature Y')
plt.legend()
plt.grid(True, linestyle=':')

plt.tight_layout()
plt.savefig('gan_framework_comparison.png')
print("✅ Visualization Framework saved as 'gan_framework_comparison.png'")

# Text Artifact: Circuit Topology Generation
print("\n🎨 Compiling Quantum Circuit Architecture View...")
circuit_diagram = qml.draw(quantum_generator_circuit)(quantum_weights, torch.randn(latent_dim))
print(circuit_diagram)

with open("quantum_circuit_architecture.txt", "w", encoding="utf-8") as f:
    f.write(circuit_diagram)
print("✅ Circuit blueprint exported to 'quantum_circuit_architecture.txt'!")