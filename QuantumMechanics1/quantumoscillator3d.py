import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.widgets as widgets
import math
import random
from mpl_toolkits.mplot3d import Axes3D

# Constants
m = 1
w = 0.5
hbar = 6.63 / (2 * np.pi)
sigma = 10

# grid
x = np.linspace(-10, 10, 1000)

# eigenstates
def psin_x(x, n):
    factor = 1 / np.sqrt(2**n * math.factorial(n))
    w_term = (m * w / (np.pi * hbar))**0.25
    gaussian = np.exp(-m * w * x**2 / (2 * hbar))
    hermite = spec.hermite(n)
    he = hermite(np.sqrt(m * w / hbar) * x)
    return factor * w_term * gaussian * he

# psi0
Å = 1e-10  # Scale for Ångstrom
psi0 = np.exp(-x**2/(sigma))/(np.sqrt(np.pi))

# Compute eigenstates
n_max = 10
eigenstates = np.array([psin_x(x, n) for n in range(n_max)])

# Compute coefficients
coeffs = np.dot(eigenstates, psi0)*1j
coeffs = list((random.random()+1)/2 for i in range(len(eigenstates)))
good_coeffs = [0.14579837952337982, 0.14639877348297736, 0.8387312297599984, 0.6256841251283494, 0.6887736718236221, 0.3023751542494041, 0.5659092051495024, 0.09043677299050601, 0.4421630452741061, 0.6228572211378776]
print("Coefficients (inner product with eigenstates):", coeffs)

# Compute superposition
def compute_superposition(x, t, coeffs, enabled_states):
    superposition = np.zeros_like(x, dtype=complex)
    for idx, n in enumerate(range(len(coeffs))):
        if enabled_states[idx]:  # Only include enabled states
            superposition += coeffs[idx] * psin_x(x, n) * np.exp(-1j * w * t*(n+0.5))
    mag = 1/np.sqrt(np.sum(superposition*np.conj(superposition)))
    return 4*superposition*mag

# Animation setup
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d', facecolor = 'black')

# Plot settings
ax.set_xlim(-10, 10)
ax.set_ylim(-1.5, 1.5)
ax.set_zlim(-1.5, 1.5)
ax.set_title("3D Wavefunction Superposition")
ax.set_xlabel("x")
ax.set_ylabel("Re(ψ)")
ax.set_zlabel("Im(ψ)")
fig.patch.set_facecolor('black')

# Checkbox setup
rax = plt.axes([0.05, 0.4, 0.15, 0.4], facecolor='lightgoldenrodyellow')
pax = plt.axes([0.05, 0.8, 0.15, 0.2], facecolor='lightgoldenrodyellow')
labels = [f"n={n}" for n in range(n_max)]
label2 = "Enable Real"
label3 = "Enable Imag"
labels2 = [label2, label3]
enabled_states = [True] * n_max
real_imag_enabled = [True]*2
check = widgets.CheckButtons(rax, labels, enabled_states)
check2 = widgets.CheckButtons(pax, labels2, real_imag_enabled)

# Toggle states
def toggle_state(label):
    idx = int(label.split("=")[1])
    enabled_states[idx] = not enabled_states[idx]

def toggle_realimag(label):
    if label == label2:
        real_imag_enabled[0] = not real_imag_enabled[0]
    if label == label3:
        real_imag_enabled[1] = not real_imag_enabled[1]
    

check.on_clicked(toggle_state)
check2.on_clicked(toggle_realimag)
fig.patch.set_facecolor('black')
ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))


# Animation function
def animate(i):
    t = i * 0.1  # Time increment
    superposition = compute_superposition(x, t, coeffs, enabled_states)
    real_part = np.real(superposition)
    imag_part = np.imag(superposition)
    
    # Clear previous plot and update the 3D plot
    ax.clear()
    
    # Redraw axis and title
    ax.set_xlim(-10, 10)
    ax.set_ylim(-1.5, 1.5)
    ax.set_zlim(-1.5, 1.5)
    ax.set_title("3D Wavefunction Superposition")
    ax.set_xlabel("x")
    ax.set_ylabel("Re(ψ)")
    ax.set_zlabel("Im(ψ)")
    
    # Plot the complex wavefunction in 3D space
    ax.plot(x, real_part, imag_part, color='white')
    if real_imag_enabled[0]:
        ax.plot(x, real_part, np.zeros_like(x), color='red')
    if real_imag_enabled[1]:
        ax.plot(x, np.zeros_like(x), imag_part, color='blue')
    
    return ax, 
# Create animation
frame = int((2*np.pi/w)*100)
anim = animation.FuncAnimation(
    fig, animate, frames=frame, interval=10, blit=False
)

plt.show()

anim.save('quantumoscillator.mp4', fps=60, extra_args=['-vcodec', 'libx264'], dpi=300)