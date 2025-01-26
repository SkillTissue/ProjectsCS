import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad_vec
import matplotlib.animation as animation

image = cv2.imread("riseandfall.png")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

ret, threshold = cv2.threshold(image_gray, 127, 255, 0)
contours, hierarch = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
contours = np.array(contours[0])

x_coords, y_coords = contours[:, :, 0].reshape(-1,), -contours[:, :, 1].reshape(-1,)

x_coords = x_coords - np.mean(x_coords)
y_coords = y_coords - np.mean(y_coords)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x_coords, y_coords)

xlim_data = plt.xlim() 
ylim_data = plt.ylim()

plt.show() #displaying the image that will be drawn

t_coords = np.linspace(0, 2*np.pi, len(x_coords)) 

num_vectors = 400

def interpolated_func(t, t_coords, x_coords, y_coords):
    return np.interp(t, t_coords, x_coords + 1j*y_coords)

coefficients = []
for n in range(-num_vectors, num_vectors+1):
    indiv_coef = 1/(2*np.pi)*quad_vec(lambda t: interpolated_func(t, t_coords, x_coords, y_coords)*np.exp(-n*t*1j), 0, 2*np.pi, limit=100, full_output=1)[0]
    coefficients.append(indiv_coef)

c = np.array(coefficients)
d_x, d_y = [], []

fig, ax = plt.subplots()

vecs = []
for i in range(-num_vectors, num_vectors+1):
    line = ax.plot([], [], 'b-', marker='o', markevery=[0, -1], markersize=2, linewidth=1)[0]
    vecs.append(line)

actual_draw, = ax.plot([], [], 'k-', linewidth=1)

ax.set_xlim(xlim_data[0]-200, xlim_data[1]+200)
ax.set_ylim(ylim_data[0]-200, ylim_data[1]+200)

ax.set_axis_off()

ax.set_aspect('equal')

writer = animation.PillowWriter(bitrate = 500, fps = 60)

frames = 600

def sorting_coefficients(coefficients):
    sorted_coeffs = []
    sorted_coeffs.append(coefficients[num_vectors])
    for i in range(1, num_vectors+1):
        sorted_coeffs.extend([coefficients[num_vectors+i],coefficients[num_vectors-i]])
    return np.array(sorted_coeffs)

def update(i, time, coefficients):

    t = time[i]

    exponential_terms = []
    for n in range(-num_vectors, num_vectors+1):
        e_term = np.exp(n * t * 1j)
        exponential_terms.append(e_term)

    coeffs = sorting_coefficients(coefficients*exponential_terms)  

    x_c = np.real(coeffs)
    y_c = np.imag(coeffs)

    center_x = 0
    center_y = 0

    for i, (x_c, y_c) in enumerate(zip(x_c, y_c)):
        p = np.linalg.norm([x_c, y_c]) 

        theta = np.linspace(0, 2*np.pi, num=50) 
        x, y = center_x + p * np.cos(theta), center_y + p * np.sin(theta)

        x, y = [center_x, center_x + x_c], [center_y, center_y + y_c]
        vecs[i].set_data(x, y)

        center_x += x_c 
        center_y += y_c
    
    d_x.append(center_x)
    d_y.append(center_y)

    actual_draw.set_data(d_x, d_y)

time = np.linspace(0, 2*np.pi, num=frames)
anim = animation.FuncAnimation(fig, update, frames=frames, fargs=(time, c),interval=5)
anim.save('riseandfall.gif', writer=writer, dpi = 300)









