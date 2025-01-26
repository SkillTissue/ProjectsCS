import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
Create Your Own Lattice Boltzmann Simulation (With Python)
Philip Mocz (2020) Princeton University, @PMocz

Simulate flow past cylinder for an isothermal fluid.
"""

def main():
    """ Lattice Boltzmann Simulation """
    # Simulation parameters
    Nx = 400    # resolution x-dir
    Ny = 100    # resolution y-dir
    rho0 = 100  # average density
    tau = 0.6   # collision timescale
    Nt = 40000   # number of timesteps
    plot_interval = 10  # Update plot every 10 iterations

    # Lattice speeds / weights
    NL = 9
    idxs = np.arange(NL)
    cxs = np.array([0, 0, 1, 1, 1, 0, -1, -1, -1])
    cys = np.array([0, 1, 1, 0, -1, -1, -1, 0, 1])
    weights = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])  # sums to 1

    # Initial Conditions
    F = np.ones((Ny, Nx, NL))  # * rho0 / NL
    np.random.seed(42)
    F += 0.01 * np.random.randn(Ny, Nx, NL)
    X, Y = np.meshgrid(range(Nx), range(Ny))
    F[:, :, 3] += 2 * (1 + 0.2 * np.cos(2 * np.pi * X / Nx * 4))
    rho = np.sum(F, 2)
    for i in idxs:
        F[:, :, i] *= rho0 / rho

    # Cylinder boundary
    cylinder = (X - Nx / 4) ** 2 + (Y - Ny / 2) ** 2 < (Ny / 4) ** 2

    # Prep figure
    fig, ax = plt.subplots(figsize=(8, 4))
    vorticity_img = ax.imshow(np.zeros((Ny, Nx)), cmap='bwr', animated=True)
    plt.colorbar(vorticity_img, ax=ax, label='Vorticity')
    ax.set_title('Vorticity Field')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    def update(it):
        nonlocal F, rho  # To modify these variables inside the function
        # Drift
        for i, cx, cy in zip(idxs, cxs, cys):
            F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

        # Reflective boundaries
        bndryF = F[cylinder, :]
        bndryF = bndryF[:, [0, 5, 6, 7, 8, 1, 2, 3, 4]]

        # Calculate fluid variables
        rho = np.sum(F, 2)
        ux = np.sum(F * cxs, 2) / rho
        uy = np.sum(F * cys, 2) / rho

        # Apply Collision
        Feq = np.zeros(F.shape)
        for i, cx, cy, w in zip(idxs, cxs, cys, weights):
            Feq[:, :, i] = rho * w * (
                1 + 3 * (cx * ux + cy * uy)
                + 9 * (cx * ux + cy * uy) ** 2 / 2
                - 3 * (ux**2 + uy**2) / 2
            )
        F += -(1.0 / tau) * (F - Feq)

        # Apply boundary
        F[cylinder, :] = bndryF

        # Compute vorticity
        vorticity = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) - (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
        vorticity[cylinder] = np.nan  # Mask vorticity inside the cylinder

        # Update plot
        vorticity_img.set_array(vorticity)
        return [vorticity_img]

    ani = FuncAnimation(fig, update, frames=range(0, Nt, plot_interval), blit=True)

    # Save animation
    ani.save('lattice_boltzmann_simulation.mp4', fps=60, writer='ffmpeg')

    plt.show()


if __name__ == "__main__":
    main()
