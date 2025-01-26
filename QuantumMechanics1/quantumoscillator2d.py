import numpy as np
import scipy.special as spec
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.widgets as widgets
import math
import random

# Constants
m = 1
w = 0.5
hbar = 6.63 / (2 * np.pi)
sigma = 16

#grid
x = np.linspace(-10, 10, 1000)

#eigenstates
def psin_x(x, n):
    factor = 1 / np.sqrt(2**n * math.factorial(n))
    w_term = (m * w / (np.pi * hbar))**0.25
    gaussian = np.exp(-m * w * x**2 / (2 * hbar))
    hermite = spec.hermite(n)
    he = hermite(np.sqrt(m * w / hbar) * x)
    return factor * w_term * gaussian * he

# psi0
Å = 1e-10  # Scale for Ångstrom
psi0 = np.exp(-x**2/(sigma))/(np.sqrt(np.pi))*np.exp(x*1j)

# Compute eigenstates
n_max = 10
eigenstates = np.array([psin_x(x, n) for n in range(n_max)])

# Compute coefficients
coeffs = np.dot(eigenstates, psi0)*1j
coeffs1 = np.ones_like(coeffs)
#coeffs = list((random.random()+1)/2 for i in range(len(eigenstates)))
good_coeffs = [0.14579837952337982, 0.14639877348297736, 0.8387312297599984, 0.6256841251283494, 0.6887736718236221, 0.3023751542494041, 0.5659092051495024, 0.09043677299050601, 0.4421630452741061, 0.6228572211378776]
print("Coefficients (inner product with eigenstates):", coeffs)

#compute superposition
def compute_superposition(x, t, coeffs, enabled_states):
    superposition = np.zeros_like(x, dtype=complex)
    for idx, n in enumerate(range(len(coeffs))):
        if enabled_states[idx]:  # Only include enabled states
            superposition += coeffs[idx] * psin_x(x, n) * np.exp(-1j * w * t*(n+0.5))
            #superposition += psin_x(x, n) * np.exp(-1j * w * t*(n+0.5))
    mag = 1/np.sqrt(np.sum(superposition*np.conj(superposition)))
    return 2*superposition*mag

#animation setup
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.25, bottom=0.25)

#plot settings
ax.set_xlim(-10, 10)
ax.set_ylim(-0.5, 1.5)
ax.set_title("Wavefunction Superpositions")
ax.set_xlabel("x")
ax.set_ylabel("ψ")
line_real, = ax.plot([], [], "b", label="Re(ψ)")
line_imag, = ax.plot([], [], "r", label="Im(ψ)")
line_abs, = ax.plot([], [], "w", label="Abs(ψ)")
ax.legend(loc=1, fontsize=8, fancybox=False)
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

#checkbox setup
rax = plt.axes([0.05, 0.4, 0.15, 0.4], facecolor='lightgoldenrodyellow')
labels = [f"n={n}" for n in range(n_max)]
enabled_states = [True] * n_max
check = widgets.CheckButtons(rax, labels, enabled_states)

#toggle states
def toggle_state(label):
    idx = int(label.split("=")[1])
    enabled_states[idx] = not enabled_states[idx]

check.on_clicked(toggle_state)

#animation function
def animate(i):
    t = i * 0.1  # Time increment
    superposition = compute_superposition(x, t, coeffs1, enabled_states)
    line_real.set_data(x, np.real(superposition))
    line_imag.set_data(x, np.imag(superposition))
    line_abs.set_data(x, 2*np.abs(superposition))
    return line_real, line_imag, line_abs,

#create animation
frame = int((2*np.pi/w)*100)
anim = animation.FuncAnimation(
    fig, animate, frames=frame, interval=10, blit=True
)

plt.show()