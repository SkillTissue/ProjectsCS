import jax.numpy as jnp
from jax import jit
import jax
import numpy as np
from diffrax import diffeqsolve, ODETerm, Tsit5
import matplotlib.pyplot as plt
import matplotlib.animation as animation
jax.config.update("jax_enable_x64", True)   

# Constants
hbar = 1.0  # Planck's constant
m = 1.0     # Mass of the particle
dx = 0.025   # Spatial step
dt = 0.01  # Time step

# Create a 2D grid
x = jnp.arange(0, 10, dx)
y = jnp.arange(0, 10, dx)
X, Y = jnp.meshgrid(x, y)

# Initial wavefunction: Gaussian wave packet
kx, ky = 10, 10  # Wave numbers
sigma = 0.5     # Width of the Gaussian
x0, y0 = 3, 3    # Initial position of the wave packet
A = 1.0 / (sigma * jnp.sqrt(jnp.pi))  # Normalization constant

psi0 = jnp.sqrt(A) * jnp.exp(-((X - x0)**2 + (Y - y0)**2) / (2 * sigma**2)) \
       * jnp.exp(1j * (kx * X + ky * Y))

# Define the potential V(x, y)
V = jnp.zeros_like(X)
barrier_mask = (X > 4.75) & (X < 5.25) & ~(((Y > 3.9) & (Y < 4.1)) | ((Y > 5.9) & (Y < 6.1)))
V = V.at[barrier_mask].set(1000)

# JAX-compatible finite difference Laplacian operator
def laplacian_2d(psi, dx):
    """Compute the 2D Laplacian using finite differences."""
    laplacian = (
        -4 * psi
        + jnp.roll(psi, shift=1, axis=0) + jnp.roll(psi, shift=-1, axis=0)
        + jnp.roll(psi, shift=1, axis=1) + jnp.roll(psi, shift=-1, axis=1)
    ) / dx**2
    return laplacian

# Time derivative of the wavefunction
@jit
def psi_t(t, y, args):
    psi = y.reshape(X.shape)  # Reshape to 2D
    laplacian_psi = laplacian_2d(psi, dx)
    psi_p = -1j * (-0.5 * hbar / m * laplacian_psi + (V / hbar) * psi)
    return psi_p

term = ODETerm(psi_t)
solver = Tsit5()

# Solve the Schrödinger equation
t_, tf = 0.0, 1.0  # Initial and final times
t_eval = jnp.arange(t_, tf, dt)  # Time steps to save
psi0_flat = psi0.flatten()  # Flatten initial wavefunction for solver

print (psi0_flat)

sol = diffeqsolve(term, solver, t0=t_, t1=tf, dt0=dt, y0=psi0_flat)

print (sol.ys)

# Reshape solution back to 2D
psi_t_all = sol.ys.reshape((len(x), len(y), -1))

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
ax.set_zlabel("|ψ(x, y, t)|²")
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.grid(False)

def update(frame):
    ax.clear()
    psi_abs2 = np.abs(psi_t_all[:, :, frame])**2  # Compute |ψ|²
    print (f"Frame: {frame}, Abs Psi max = {jnp.max(psi_abs2)}")
    potential_surface = ax.plot_surface(
        X, Y, V * 0.00001, cmap="inferno", edgecolor='none', alpha=0.5, label="Potential"
    )
    ax.plot_surface(X, Y, psi_abs2, cmap="viridis", edgecolor='none')
    ax.set_title(f"Time = {t_eval[frame]:.3f}")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 0.1)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("|ψ(x, y, t)|²")

ani = animation.FuncAnimation(fig, update, frames=len(t_eval), interval=50)
ani.save("schrodinger_2d_ydse_opt.mp4", fps=60)
plt.show()
