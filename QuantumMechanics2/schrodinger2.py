import numpy as np
import plotly.graph_objects as go
from scipy.special import genlaguerre, sph_harm
import math

# Define functions for wavefunction and color mapping
a0 = 1  # Bohr radius

def xyz_to_rtp(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    t = np.arccos(z/r)
    p = np.sign(y)*np.arccos(x/np.sqrt(x**2 + y**2))
    return r, t, p

def laguerre(n, l, r):
    rho = 2 * r / (n * a0)
    return genlaguerre(n - l - 1, 2 * l + 1)(rho)

def Ye(m, l, theta, phi):
    return sph_harm(m, l, phi, theta)

def psi_abs(x, y, z, n, l, m):
    r, theta, phi = xyz_to_rtp(x, y, z)
    prefactor = np.sqrt((2 / (n * a0))**3 * math.factorial(n - l - 1) /
                        (2 * n * math.factorial(n + l)))
    radial_part = np.exp(-r / (n * a0)) * (2 * r / (n * a0))**l * laguerre(n, l, r)
    angular_part = Ye(m, l, theta, phi)
    return 10000*np.abs(prefactor * radial_part * angular_part)**2

# Grid setup
n, l, m = 4, 3, 1  # Quantum numbers
grid_size = 50
L = 100
x = np.linspace(-L, L, grid_size)
y = np.linspace(-L, L, grid_size)
z = np.linspace(-L, L, grid_size)

X, Y, Z = np.meshgrid(x, y, z)
prob_densities = np.zeros_like(X)

# Calculate probability densities for each point
prob_densities = psi_abs(X, Y, Z, n, l, m)
prob1 = np.reshape(prob_densities, (-1))/(4*1.234)
print ("total probability: ", sum(prob1))
print ("max probability: ", np.max(prob1))
print ('average: ', np.average(prob1))
prob = np.average(prob1)
threshold = 0.01
prob1 = [prob_d if np.abs(prob_d - prob) < threshold else 0 for prob_d in prob1]
prob1 = np.reshape(prob1, (50, 50, 50))

# Create an isosurface from the probability density
isosurface_value = np.max(prob1)*0.01 # Set isosurface at 5% of max density

# Plotly 3D isosurface
fig = go.Figure(data=go.Isosurface(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=prob1.flatten(),
    isomin=isosurface_value,
    isomax=np.max(prob1),
    opacity=0.3,
    surface_count=10,  # Number of isosurfaces to display
    colorscale='Viridis',  # Choose a colorscale
))

# Layout customization
fig.update_layout(
    scene=dict(
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        zaxis=dict(showgrid=False, zeroline=False, visible=False),
    ),
    title=dict(
        text=f'Probability Density for n={n}, l={l}, m={m}',
        font=dict(color="white")
    ),
    paper_bgcolor="black",  # Set plot area background to black
    scene_aspectmode="cube",  # Make axes equal for better visualization
)

# Show the plot
fig.show()
