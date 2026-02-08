from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np

def animate_kiiking(model, interval=10, shadow_frame=50):

    """
    Create animate output  of kiiking pendulum.
    blue line
    interval = 10 millisec - matches dt 0.1, so is in 'realtime'
    -> this means the shadow frame is last 0.5s
    TODO add timer in corner of animation pot to have time
    """

    if model.theta is None:
        raise ValueError("Model must be solved before animating")

    # Downsample for animation performance
    theta = model.theta
    r = model.r

    # Cartesian coordinates
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    # Infer frame interval if not provided
    if interval is None:
        dt = model.t[1] - model.t[0]
        interval = int(1000 * dt )

    fig, ax = plt.subplots(figsize=(6, 6))

    # Stable axis limits
    r_max = np.max(r)
    ax.set_xlim(-1.3 * r_max, 1.3 * r_max)
    ax.set_ylim(-1.3 * r_max, 1.3 * r_max)
    ax.set_aspect('equal')
    ax.invert_yaxis()

    ax.set_xlabel("Horizontal")
    ax.set_ylabel("Vertical (downwards)")
    ax.set_title("Kiiking Pendulum")

    # mark (0.0) as pivot
    ax.plot(0, 0, 'ko', markersize=15, zorder=1)

    # empty
    rod, = ax.plot([], [], 'o-', lw=3,
                   color='darkblue',
                   markerfacecolor='red',
                   markeredgecolor='darkred')

    trail, = ax.plot([], [], '-', lw=1.5, alpha=0.4)

    def init():
        rod.set_data([], [])
        trail.set_data([], [])
        return rod, trail

    def update(frame):
        rod.set_data([0, y[frame]], [0, x[frame]])

        start = max(0, frame - shadow_frame)
        trail.set_data(y[start:frame+1], x[start:frame+1])

        return rod, trail

    anim = FuncAnimation(
        fig,
        update,
        frames=len(theta),
        init_func=init,
        interval=interval,
        blit=True
    )

    return anim


