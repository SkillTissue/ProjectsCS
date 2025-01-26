import numpy as np
from scipy import integrate
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D

# Constants
hbar = 1.0  # Planck's constant
m = 1.0     # Mass of the particle
dx = 0.05   # Spatial step
dt = 0.01  # Time step

# Create a 2D grid
x = np.arange(0, 10, dx)
y = np.arange(0, 10, dx)
X, Y = np.meshgrid(x, y)

# Initial wavefunction: Gaussian wave packet
kx, ky = 10, 0  # Wave numbers  
sigma = 0.5     # Width of the Gaussian
x0, y0 = 1, 5    # Initial position of the wave packet
A = 1.0 / (sigma * np.sqrt(np.pi))  # Normalization constant

psi0 = np.sqrt(A) * np.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)) \
       * np.exp(1j * (kx * X + ky * Y))

# Define the potential V(x, y)
V = np.zeros_like(X)
barrier_mask = (X > 4.75) & (X < 5.25) & ~(((Y > 3.5) & (Y < 4.5)) | ((Y > 5.5) & (Y < 6.5)))
V[barrier_mask] = 1000

# Finite difference Laplacian operator in 2D
N = x.size  # Number of grid points along one axis
Laplacian = (
    sparse.diags([1, -2, 1], [-1, 0, 1], shape=(N, N)) / dx**2
)
I = sparse.identity(N)
L2D = sparse.kron(Laplacian, I) + sparse.kron(I, Laplacian)  # 2D Laplacian

# Time derivative of the wavefunction
def psi_t(t, psi_flat):
    psi = psi_flat.reshape(X.shape)  # Reshape to 2D
    laplacian_psi = L2D.dot(psi_flat)
    dpsi_dt = -1j * (-0.5 * hbar / m * laplacian_psi + (V.flatten() / hbar) * psi_flat)
    return dpsi_dt

# Solve the Schrödinger equation using `solve_ivp`
t0, tf = 0.0, 3.0  # Initial and final times
t_eval = np.arange(t0, tf, dt)  # Time steps to save
psi0_flat = psi0.flatten()  # Flatten initial wavefunction for solver

sol = integrate.solve_ivp(
    psi_t, t_span=(t0, tf), y0=psi0_flat, t_eval=t_eval, method='RK23'
)

# Reshape solution back to 2D
psi_t_all = sol.y.reshape((len(x), len(y), -1))

# Plot the potential for reference
plt.figure(figsize=(6, 6))
plt.contourf(X, Y, V, levels=20, cmap="inferno")
plt.colorbar(label="Potential $V(x, y)$")
plt.title("Potential Landscape")
plt.xlabel("x")
plt.ylabel("y")
plt.show()


# Animate the wavefunction |psi(x, y, t)|^2
fig = plt.figure(figsize=(8, 6), facecolor = 'black')
ax = fig.add_subplot(111, projection='3d', facecolor = 'black')
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(0, 0.1)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("|ψ(x, y, t)|² * 0.15")
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.grid(False)
ax.plot_surface(X, Y, V)

def update(frame):
    ax.clear()
    print ("Frame: ", frame)
    psi_abs2 = np.abs(psi_t_all[:, :, frame])**2  # Compute |ψ|²
    potential_surface = ax.plot_surface(
        X, Y, V * 0.0001, cmap="inferno", edgecolor='none', alpha=0.25, label="Potential"
    )
    ax.plot_surface(X, Y, psi_abs2*0.15, cmap="viridis", edgecolor='none')
    ax.set_title(f"Time = {t_eval[frame]:.3f}")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("|ψ(x, y, t)|²")

ani = animation.FuncAnimation(fig, update, frames=len(t_eval), interval=50)
ani.save("schrodinger_2d_ydse_new.mp4", fps=60)
plt.show()
