import numpy as np
import matplotlib.pyplot as plt
from scipy.special import genlaguerre, sph_harm

a0 = 1  # Bohr radius

def laguerre(n, l, r):
    rho = 2 * r / (n * a0)
    return genlaguerre(n - l - 1, 2 * l + 1)(rho)

def Ye(m, l, theta, phi):
    return sph_harm(m, l, phi, theta)

def psi(r, theta, phi, n, l, m):
    prefactor = np.sqrt((2 / (n * a0))**3 * np.math.factorial(n - l - 1) /
                        (2 * n * np.math.factorial(n + l)))
    radial_part = np.exp(-r / (n * a0)) * (2 * r / (n * a0))**l * laguerre(n, l, r)
    angular_part = Ye(m, l, theta, phi)
    return prefactor * radial_part * angular_part

# Spherical coordinate grid
N = 25
phi = np.linspace(0, 2 * np.pi, N)
theta = np.linspace(0, np.pi, N)
r = np.linspace(0, 20, N)
r, phi, theta = np.meshgrid(r, phi, theta, indexing='ij')

n, l, m = 4, 3, 1  # Quantum numbers
threshold = 0.45  # Probability density threshold

# Compute the probability density
prob = 10000*np.abs(psi(r, theta, phi, n, l, m))**2

# Find the combinations where probability density is close to the threshold
tolerance = 0.01
matching_indices = np.where(np.abs(prob - threshold) <= tolerance)

# Extract corresponding r, theta, and phi values
matching_r = r[matching_indices]
matching_theta = theta[matching_indices]
matching_phi = phi[matching_indices]

X = matching_r * np.sin(matching_theta) * np.cos(matching_phi)
Y = matching_r * np.sin(matching_theta) * np.sin(matching_phi)
Z = matching_r * np.cos(matching_theta)

# Visualize the matching points in 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(matching_r * np.sin(matching_theta) * np.cos(matching_phi),
           matching_r * np.sin(matching_theta) * np.sin(matching_phi),
           matching_r * np.cos(matching_theta), label=f'Probability ~ {threshold}')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title(f"Hydrogen Wavefunction Matches: n={n}, l={l}, m={m}")
plt.show()
