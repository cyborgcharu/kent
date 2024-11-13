import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_modified_spacetime_curvature(x, y, mass, λ_gradient):
    """
    Calculate spacetime curvature with attention modification
    """
    G = 1  # Gravitational constant (natural units)
    
    # Standard gravitational potential
    r = np.sqrt(x**2 + y**2)
    V_standard = -G * mass / (r + 1e-10)  # Avoid division by zero
    
    # Attention modification
    attention_term = np.exp(-λ_gradient * r)
    V_modified = V_standard * (1 + λ_gradient * attention_term)
    
    return V_modified

def plot_modified_spacetime(mass=1.0, λ_values=[0, 1, 5]):
    # Create coordinate grid
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    
    fig = plt.figure(figsize=(15, 5))
    
    for i, λ in enumerate(λ_values):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Calculate curvature
        Z = calculate_modified_spacetime_curvature(X, Y, mass, λ)
        
        # Plot surface
        surf = ax.plot_surface(X, Y, Z, cmap='viridis')
        
        ax.set_title(f'λ = {λ}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('Curvature')
        
        # Add colorbar
        fig.colorbar(surf, ax=ax)
    
    plt.suptitle('Spacetime Curvature Modified by Attention Gradient')
    plt.tight_layout()
    plt.show()

# Calculate gravitational lensing modification
def calculate_light_deflection(impact_parameter, mass, λ_gradient):
    """
    Calculate modified light deflection angle
    """
    # Standard deflection angle
    α_standard = 4 * mass / impact_parameter
    
    # Attention modification
    attention_term = np.exp(-λ_gradient * impact_parameter)
    α_modified = α_standard * (1 + λ_gradient * attention_term)
    
    return α_standard, α_modified

# Plot deflection angles
def plot_deflection_comparison():
    impact_parameters = np.linspace(1, 10, 100)
    mass = 1.0
    λ_values = [0, 1, 2, 5]
    
    plt.figure(figsize=(10, 6))
    
    for λ in λ_values:
        α_std, α_mod = calculate_light_deflection(impact_parameters, mass, λ)
        plt.plot(impact_parameters, α_mod, label=f'λ = {λ}')
    
    plt.plot(impact_parameters, α_std, '--', label='Standard GR', color='black')
    plt.title('Modified Light Deflection vs Impact Parameter')
    plt.xlabel('Impact Parameter')
    plt.ylabel('Deflection Angle')
    plt.legend()
    plt.grid(True)
    plt.show()

# Run simulations
print("Plotting spacetime curvature...")
plot_modified_spacetime()

print("\nPlotting light deflection...")
plot_deflection_comparison()

# Calculate some specific numerical predictions
b_test = 5.0  # test impact parameter
m_test = 1.0  # test mass
λ_test = 1.0  # test attention gradient

α_standard, α_modified = calculate_light_deflection(b_test, m_test, λ_test)
print(f"\nNumerical Predictions for b={b_test}, M={m_test}, λ={λ_test}:")
print(f"Standard Deflection: {α_standard:.4f}")
print(f"Modified Deflection: {α_modified:.4f}")
print(f"Attention Effect: {((α_modified/α_standard - 1)*100):.1f}%")