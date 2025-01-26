from manim import *
import numpy as np
from scipy.integrate import solve_ivp

# Lorenz system definition
def lorenz_system(t, state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


def ode_solution(function, state0, time, dt=0.01):
    t_eval = np.arange(0, time, dt)
    solution = solve_ivp(
        function, t_span=(0, time), y0=state0, t_eval=t_eval, args=()
    )
    return np.array(solution.y).T  

# Manim Scene
class LorenzAttractor(ThreeDScene):
    def construct(self):

        axes = ThreeDAxes(
            x_range=(-50, 50),
            y_range=(-50, 50),
            z_range=(-50, 50)
        )
        axes.center()

        self.set_camera_orientation(phi=60 * DEGREES, theta=30 * DEGREES)

        self.play(FadeIn(axes))

        self.begin_ambient_camera_rotation(rate=1, about='theta')
        
        states = [
            [10, 10, 10 + n*0.01]
            for n in range(3)
        ]
        time = 30
        dt = 0.01

        colors = [BLUE, TEAL, RED]

        animations = []

        curves = VGroup()
        for state, color in zip(states, colors):
            points = ode_solution(lorenz_system, state, time, dt)
            manim_points = [axes.c2p(x,y,z) for x, y, z in points]
            curve = VMobject().set_points_smoothly(manim_points)
            curve.set_stroke(color, 2)
            curves.add(curve)
            animations.append(Create(curve, run_time=30, rate_functions = linear))


        dots = Group(*[Dot3D(color=color, point = axes.c2p(0, 0, 0), radius = 0.02) for color in colors])

        def update_dots(dots):
            for dot, curve in zip(dots, curves):
                dot.move_to(curve.get_end())

        dots.add_updater(update_dots)
        self.add(dots)
        
        self.play(
            AnimationGroup(
                *animations,
            ))
    

        self.wait()
