import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy
import seaborn as sns

# Part 1: Double-Slit Experiment with Attention Gradient
def simulate_double_slit(λ_gradient, num_points=1000):
    """
    Simulates double-slit experiment with attention modification
    """
    # Setup parameters
    x = np.linspace(-10, 10, num_points)
    k = 2.0  # wavenumber
    d = 2.0  # slit separation
    
    # Standard quantum interference
    psi1 = np.exp(1j * k * np.sqrt(x**2 + d**2))
    psi2 = np.exp(1j * k * np.sqrt((x + d)**2 + d**2))
    psi_standard = (psi1 + psi2) / np.sqrt(2)
    
    # Attention modification
    attention_term = np.exp(-λ_gradient * np.abs(x))
    psi_modified = psi_standard * attention_term
    
    return x, np.abs(psi_standard)**2, np.abs(psi_modified)**2

def measure_information_preservation(λ_gradient, state_size=100):
    """
    Measures how much quantum information survives under attention gradient
    """
    # Create initial quantum state - FIXED VERSION
    real_part = np.random.rand(state_size)
    imag_part = np.random.rand(state_size)
    psi = real_part + 1j * imag_part
    psi = psi / np.linalg.norm(psi)
    
    # Apply attention gradient
    attention = np.exp(-λ_gradient * np.arange(state_size))
    psi_modified = psi * attention
    psi_modified = psi_modified / np.linalg.norm(psi_modified)
    
    # Calculate von Neumann entropy
    rho = np.outer(psi, psi.conj())
    rho_modified = np.outer(psi_modified, psi_modified.conj())
    
    eigenvals = np.real(np.linalg.eigvals(rho))
    eigenvals_mod = np.real(np.linalg.eigvals(rho_modified))
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    S = -np.sum(eigenvals * np.log2(eigenvals + epsilon))
    S_modified = -np.sum(eigenvals_mod * np.log2(eigenvals_mod + epsilon))
    
    return S, S_modified

# Test different attention gradients
λ_values = [0.1, 0.5, 1.0, 2.0, 5.0]
plt.figure(figsize=(15, 8))

for λ in λ_values:
    x, standard, modified = simulate_double_slit(λ)
    plt.plot(x, modified, label=f'λ = {λ}')

plt.title('Double-Slit Interference with Attention Gradient')
plt.xlabel('Position')
plt.ylabel('Probability Density')
plt.legend()
plt.grid(True)
plt.show()

# Test information preservation
λ_range = np.linspace(0.1, 5, 20)
S_original = []
S_modified = []

for λ in λ_range:
    s_orig, s_mod = measure_information_preservation(λ)
    S_original.append(s_orig)
    S_modified.append(s_mod)

plt.figure(figsize=(12, 6))
plt.plot(λ_range, S_original, label='Original Entropy')
plt.plot(λ_range, S_modified, label='Modified Entropy')
plt.title('Quantum Information Preservation under Attention Gradient')
plt.xlabel('Attention Gradient λ')
plt.ylabel('von Neumann Entropy')
plt.legend()
plt.grid(True)
plt.show()

# Part 3: Phase Transition Analysis
def analyze_phase_transition(λ_range=np.linspace(0.1, 10, 50)):
    """
    Analyzes phase transition between quantum and classical regimes
    """
    quantum_measure = []
    classical_measure = []
    
    for λ in λ_range:
        # Quantum coherence measure
        _, _, modified = simulate_double_slit(λ)
        coherence = np.max(modified) - np.min(modified)
        quantum_measure.append(coherence)
        
        # Classical probability measure
        classical = 1 - np.exp(-λ)
        classical_measure.append(classical)
    
    return λ_range, quantum_measure, classical_measure

λ_range, quantum, classical = analyze_phase_transition()

plt.figure(figsize=(12, 6))
plt.plot(λ_range, quantum, label='Quantum Coherence')
plt.plot(λ_range, classical, label='Classical Behavior')
plt.axvline(x=1.0, color='r', linestyle='--', label='Critical Point')
plt.title('Quantum-Classical Phase Transition')
plt.xlabel('Attention Gradient λ')
plt.ylabel('Measure Strength')
plt.legend()
plt.grid(True)
plt.show()