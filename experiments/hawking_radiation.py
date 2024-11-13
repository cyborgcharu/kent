import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy

def predict_hawking_spectrum(black_hole_mass, λ_gradient, r_observer):
    # Constants (in natural units where c = G = ℏ = k_B = 1)
    M_sun = 2e30  # Solar mass in kg
    
    # Convert mass to natural units
    M = black_hole_mass * M_sun
    
    # Hawking temperature (in natural units)
    T_H = 1/(8*np.pi*M)
    
    # Modified temperature
    T_modified = T_H * (1 + λ_gradient * 2*M/r_observer)
    
    # Use dimensionless frequency
    x = np.linspace(0.1, 10, 1000)  # x = ω/T_modified
    
    # Spectrum calculation (avoiding divide by zero)
    N_omega = 1/(np.exp(x) - 1)
    
    # Attention modification
    A_omega = np.exp(-λ_gradient * x)
    
    # Modified spectrum
    N_modified = N_omega * A_omega
    
    return x, N_modified, T_modified

# Test parameters
masses = [1, 10, 100]  # solar masses
λ_values = [0.1, 1.0, 5.0]
r_obs = 10

plt.figure(figsize=(15, 6))

# Plot 1: Radiation Spectrum
plt.subplot(1, 2, 1)
for mass in masses:
    for λ in λ_values:
        x, N_mod, T = predict_hawking_spectrum(mass, λ, r_obs)
        plt.plot(x, N_mod, label=f'M={mass}M☉, λ={λ}')

plt.title('Modified Hawking Radiation Spectrum')
plt.xlabel('Dimensionless Frequency (ω/T)')
plt.ylabel('Particle Number')
plt.legend(fontsize=8)
plt.grid(True)

# Plot 2: Information Content
plt.subplot(1, 2, 2)
λ_range = np.linspace(0.1, 5, 20)
info_ratios = []

for λ in λ_range:
    _, N_mod, _ = predict_hawking_spectrum(10, λ, r_obs)
    # Calculate information ratio directly from spectrum
    S_thermal = -np.sum(N_mod * np.log(N_mod + 1e-10))
    S_modified = -np.sum(N_mod * np.exp(-λ) * np.log(N_mod * np.exp(-λ) + 1e-10))
    info_ratios.append(S_modified/S_thermal)

plt.plot(λ_range, info_ratios)
plt.title('Information Preservation vs Attention Gradient')
plt.xlabel('Attention Gradient λ')
plt.ylabel('Modified/Thermal Entropy Ratio')
plt.grid(True)

plt.tight_layout()
plt.show()

# Print some numerical results
test_mass = 10
test_λ = 1.0
_, N_mod, T_mod = predict_hawking_spectrum(test_mass, test_λ, r_obs)
print(f"\nFor M = {test_mass}M☉, λ = {test_λ}:")
print(f"Modified Temperature (natural units): {T_mod:.2e}")
S_ratio = -np.sum(N_mod * np.exp(-test_λ) * np.log(N_mod * np.exp(-test_λ) + 1e-10)) / -np.sum(N_mod * np.log(N_mod + 1e-10))
print(f"Information Preservation Ratio: {S_ratio:.2%}")