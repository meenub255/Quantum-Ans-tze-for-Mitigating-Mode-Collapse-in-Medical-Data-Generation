import torch
import torch.nn as nn
import torch.optim as optim
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt

# Set random seeds for clinical dataset reproducibility
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# 1. BIOLOGICAL DATA DISTRIBUTION (CARDIOVASCULAR TELEMETRY)
# =====================================================================
def get_real_cardio_data(batch_size):
    # Centered near a high-stress, borderline hypertensive crisis patient profile
    mean = torch.tensor([0.75, 0.65])
    covariance = torch.tensor([[0.012, 0.007], 
                               [0.007, 0.015]])
    
    m = torch.distributions.MultivariateNormal(mean, covariance)
    data = m.sample((batch_size,))
    return torch.clamp(data, -1.0, 1.0) 

# Pipeline Hyperparameters
batch_size = 16
epochs = 150       # Slightly increased for smoother convergence
latent_dim = 2

# =====================================================================
# 2. PARAMETERIZED QUANTUM CIRCUIT (PQC) GENERATOR
# =====================================================================
num_qubits = 2
dev = qml.device("default.qubit", wires=num_qubits)

@qml.qnode(dev, interface="torch")
def quantum_generator_circuit(weights, inputs):
    qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation='X')
    
    # Enhanced Ansätz Layer 1: Three-axis rotation for full Bloch Sphere expressibility
    qml.RX(weights[0], wires=0)
    qml.RY(weights[1], wires=1)
    qml.RZ(weights[2], wires=0) # Added Z-rotation for deeper phase-space control
    qml.CNOT(wires=[0, 1]) 
    
    # Enhanced Ansätz Layer 2
    qml.RY(weights[3], wires=0)
    qml.RX(weights[4], wires=1)
    qml.RZ(weights[5], wires=1)
    qml.CNOT(wires=[1, 0])
    
    return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

class ClinicalDiscriminator(nn.Module):
    def __init__(self):
        super(ClinicalDiscriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 8),
            nn.LeakyReLU(0.2),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# =====================================================================
# 3. CLASSICAL GENERATOR BASELINE
# =====================================================================
class ClassicalGenerator(nn.Module):
    def __init__(self):
        super(ClassicalGenerator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 2),
            nn.Tanh()
        )
    def forward(self, x):
        return self.model(x)

# =====================================================================
# 4. EXPERIMENTAL HYBRID TRAINING LOOP WITH LABEL SMOOTHING
# =====================================================================

# --- Train Hybrid Quantum GAN ---
print("⚡ [1/2] Training Optimized Quantum GAN on Cardiovascular Telemetry...")
q_disc = ClinicalDiscriminator()
q_weights = torch.randn(6, requires_grad=True) # Expanded to 6 weights for deep topology

# Balanced Learning Rates: Slow down discriminator so generator can keep up
qd_opt = optim.Adam(q_disc.parameters(), lr=0.005) 
qg_opt = optim.Adam([q_weights], lr=0.02)
loss_fn = nn.BCELoss()

q_g_loss_history = []

for epoch in range(epochs):
    qd_opt.zero_grad()
    real_vitals = get_real_cardio_data(batch_size)
    
    # --- IMPLEMENTING TARGET LABEL SMOOTHING (Wow factor optimization) ---
    real_labels = torch.full((batch_size, 1), 0.9) # Smooth real labels to 0.9
    fake_labels = torch.full((batch_size, 1), 0.1) # Smooth fake labels to 0.1
    
    qd_real_loss = loss_fn(q_disc(real_vitals), real_labels)
    
    noise = torch.randn(batch_size, latent_dim)
    q_fake_vitals = torch.zeros(batch_size, 2)
    for i in range(batch_size):
        q_fake_vitals[i] = torch.stack(quantum_generator_circuit(q_weights, noise[i]))
        
    qd_fake_loss = loss_fn(q_disc(q_fake_vitals.detach()), fake_labels)
    qd_loss = qd_real_loss + qd_fake_loss
    qd_loss.backward()
    qd_opt.step()
    
    qg_opt.zero_grad()
    # Generator wants discriminator to output 0.9 (smooth real)
    qg_loss = loss_fn(q_disc(q_fake_vitals), torch.full((batch_size, 1), 0.9)) 
    qg_loss.backward()
    qg_opt.step()
    q_g_loss_history.append(qg_loss.item())

# --- Train Classical GAN Baseline ---
print("⚙️ [2/2] Training Classical GAN Baseline...")
c_gen = ClassicalGenerator()
c_disc = ClinicalDiscriminator()

cd_opt = optim.Adam(c_disc.parameters(), lr=0.005)
cg_opt = optim.Adam(c_gen.parameters(), lr=0.02)

c_g_loss_history = []

for epoch in range(epochs):
    cd_opt.zero_grad()
    real_vitals = get_real_cardio_data(batch_size)
    real_labels = torch.full((batch_size, 1), 0.9)
    fake_labels = torch.full((batch_size, 1), 0.1)
    
    cd_real_loss = loss_fn(c_disc(real_vitals), real_labels)
    noise = torch.randn(batch_size, latent_dim)
    c_fake_vitals = c_gen(noise)
    cd_fake_loss = loss_fn(c_disc(c_fake_vitals.detach()), fake_labels)
    cd_loss = cd_real_loss + cd_fake_loss
    cd_loss.backward()
    cd_opt.step()
    
    cg_opt.zero_grad()
    cg_loss = loss_fn(c_disc(c_fake_vitals), torch.full((batch_size, 1), 0.9))
    cg_loss.backward()
    cg_opt.step()
    c_g_loss_history.append(cg_loss.item())

# =====================================================================
# 5. HIGH-QUALITY EXPERIMENTAL CHART COMPILATION
# =====================================================================
print("\n📊 Compiling Enhanced Visualizations...")

plt.figure(figsize=(14, 6))

# Panel 1: Stable Convergence Analysis
plt.subplot(1, 2, 1)
plt.plot(q_g_loss_history, label='Quantum Generator (PQC Loss)', color='#ff7f0e', linewidth=2)
plt.plot(c_g_loss_history, label='Classical Generator (Linear Baseline)', color='#1f77b4', linestyle='--')
plt.title('Stabilized Clinical Generator Loss Convergence', fontsize=12, fontweight='bold')
plt.xlabel('Training Epochs')
plt.ylabel('Binary Cross-Entropy Loss')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Panel 2: Optimized Clinical Feature Mapping Topology
with torch.no_grad():
    eval_noise = torch.randn(300, latent_dim)
    real_distribution = get_real_cardio_data(300)
    
    quantum_generated = torch.zeros(300, 2)
    for i in range(300):
        quantum_generated[i] = torch.stack(quantum_generator_circuit(q_weights, eval_noise[i]))
        
    classical_generated = c_gen(eval_noise)

plt.subplot(1, 2, 2)
plt.scatter(real_distribution[:, 0], real_distribution[:, 1], alpha=0.4, label='Real Patient Stress States', color='gray', s=20)
plt.scatter(quantum_generated[:, 0], quantum_generated[:, 1], alpha=0.7, label='QGAN Synthesized Vitals (Optimized)', color='#17becf', edgecolors='none', s=35)
plt.scatter(classical_generated[:, 0], classical_generated[:, 1], alpha=0.6, label='Classical Baseline Model', color='#1f77b4', marker='x', s=30)
plt.title('Optimized Hemodynamic Feature Space Topology', fontsize=12, fontweight='bold')
plt.xlabel('Normalized Systolic Blood Pressure')
plt.ylabel('Myocardial Oxygen Demand Index')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('quantum_cardio_stress_mapping_optimized.png', dpi=300)
print("✅ Output saved successfully as 'quantum_cardio_stress_mapping_optimized.png'!")