import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Affine2D
import matplotlib.animation as animation


g = 9.81
k = 20
m = 1
ell_0 = 1

x0 = np.array([np.deg2rad(45), 0, 0.4, 1])

def spring_mass_ODE(t, y):
    theta = y[0]
    theta_dot = y[1]
    ell = y[2]
    ell_dot = y[3]

    return (
        theta_dot,
        (-2.0*ell_dot*theta_dot - g*np.sin(theta))/(ell + ell_0),
        ell_dot,
        -ell*k/m + ell*theta_dot**2 + ell_0*theta_dot**2 + g*np.cos(theta),
    )

sol = solve_ivp(spring_mass_ODE, [0, 10], x0, 
    t_eval=np.linspace(0,10,10*60))

theta = sol.y[0]
ell = sol.y[2]
t = sol.t

fig = plt.figure()
ax = fig.add_subplot(aspect='equal')

def generate_spring(n):
    data = np.zeros((2,n+2)) 
    data[:,-1] = [0,-1]
    for i in range(1,n+1):
        data[0,i] = -1/(2*n) if i % 2 else 1/(2*n)
        data[1,i] = -(2*i-1)/(2*n)
    return data

fig = plt.figure()
ax = fig.add_subplot(aspect='equal')
fig.patch.set_facecolor('black')
ax.set_facecolor('black')
ax.set_xlim(-2, 2)
ax.set_ylim(-3, 0.7)

ell_0 = 1

data = ell_0*np.append(generate_spring(60), np.ones((1,60+2)), axis=0)
spring = Line2D(data[0,:], data[1,:], color='w', linewidth = 0.1)
circle = ax.add_patch(plt.Circle((0,0), 0.1, fc='w', zorder=3)) 
ax.add_line(spring)

def animate(i):

    spring_length = (ell_0+ell[i])
    px = spring_length* np.sin(theta[i])
    py = -spring_length * np.cos(theta[i])
    circle.set_center((px, py))
    

    A = Affine2D().scale(2/spring_length, spring_length).rotate(theta[i]).get_matrix()
    data_new = A @ data
    spring.set_data(data_new[0,:], data_new[1,:])


ani = animation.FuncAnimation(fig, animate, frames=len(t))
writer = animation.FFMpegWriter(bitrate = 1000, fps = 60)
ani.save('elastic_pendulum.mp4', writer= writer, dpi= 300)



