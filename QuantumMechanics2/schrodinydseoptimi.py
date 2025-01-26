import jax.numpy as jnp
from jax import jit
import jax
from diffrax import diffeqsolve, ODETerm, Tsit5
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

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

# Time derivative of the real and imaginary parts of the wavefunction
@jit
def psi_re_im(t, y, args):
    psi_re, psi_im = y.reshape(2, X.shape[0], X.shape[1])  # Reshape to 2D

    laplacian_re = laplacian_2d(psi_re, dx)
    laplacian_im = laplacian_2d(psi_im, dx)

    # Equation for real part
    dpsi_re = - (hbar / (2 * m)) * laplacian_im + (V / hbar) * psi_im

    # Equation for imaginary part
    dpsi_im = (hbar / (2 * m)) * laplacian_re - (V / hbar) * psi_re

    return jnp.concatenate([dpsi_re.flatten(), dpsi_im.flatten()])

term = ODETerm(psi_re_im)
solver = Tsit5()

# Solve the Schrödinger equation
t_, tf = 0.0, 1.0  # Initial and final times
t_eval = jnp.arange(t_, tf, dt)  # Time steps to save

# Initial condition for the real and imaginary parts
psi0_flat = jnp.concatenate([jnp.real(psi0).flatten(), jnp.imag(psi0).flatten()])

# Solve the system
sol = diffeqsolve(term, solver, t0=t_, t1=tf, dt0=dt, y0=psi0_flat)

# Reshape solution back to 2D (real and imaginary parts)
psi_re_im_all = sol.ys.reshape((len(t_eval), 2, len(x), len(y)))

# Plot the potential for reference
plt.figure(figsize=(6, 6))
plt.contourf(X, Y, V, levels=20, cmap="inferno")
plt.colorbar(label="Potential $V(x, y)$")
plt.title("Potential Landscape")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# Animate the wavefunction |psi(x, y, t)|²
fig = plt.figure(figsize=(8, 6), facecolor='black')
ax = fig.add_subplot(111, projection='3d', facecolor='black')
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
    psi_re = psi_re_im_all[frame, 0, :, :]
    psi_im = psi_re_im_all[frame, 1, :, :]
    psi_abs2 = np.abs(psi_re + 1j * psi_im)**2  # Compute |ψ|²

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
ani.save("schrodinger_2d_split_re_im.mp4", fps=60)
plt.show()
