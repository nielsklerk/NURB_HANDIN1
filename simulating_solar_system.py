# imports
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # REMOVE

from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric_posvel
import astropy.units as u

from matplotlib.animation import FuncAnimation
from matplotlib import animation

G = 4.0 * np.pi**2  # in AU^3 / (M_sun yr^2)

bodies_with_masses = {
    "sun": 1.0,
    "mercury": 1.660e-7,
    "venus": 2.447e-6,
    "earth": 3.003e-6,
    "mars": 3.227e-7,
    "jupiter": 9.545e-4,
    "saturn": 2.858e-4,
    "uranus": 4.366e-5,
    "neptune": 5.151e-5,
}
# in M_sun

# Question 1: Simulating the solar system



def get_initial_conditions(time: Time) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate initial conditions for the Solar System using get_body_barycentric_posvel.

    Positions are in AU, while velocities are in AU/yr.

    Parameters
    ----------
    time : astropy.time.Time
        Time at which the initial conditions are evaluated.

    Returns
    -------
    positions : ndarray
        Initial positions, shape (N_bodies, 3).
    velocities : ndarray
        Initial velocities, shape (N_bodies, 3).
    """
    positions = np.zeros((len(bodies_with_masses), 3))
    velocities = np.zeros((len(bodies_with_masses), 3))

    # Loop over the Sun and the planets
    for i, planet in enumerate(bodies_with_masses.keys()):

        # Load the position and velocity of the Sun the planets
        with solar_system_ephemeris.set("jpl"):
            position, velocity = get_body_barycentric_posvel(planet, time)
        
        # Convert position and velicoty to units of AU and AU/day
        positions[i] = position.xyz.to_value(u.AU)
        velocities[i] = velocity.xyz.to_value(u.AU / u.yr)
    
    return positions, velocities


def compute_accelerations(
    positions: np.ndarray,
    masses: np.ndarray
) -> np.ndarray:
    """
    Compute the accelerations using equation 1 in the hand in.

    Parameters
    ----------
    positions : ndarray
        Positions of all bodies, shape (N_bodies, 3)
    masses : ndarray
        Masses of all bodies, shape (N_bodies,)

    Returns
    -------
    accelerations : ndarray
        Accelerations of all bodies, shape (N_bodies, 3)
    """
    def norm(x):
        return np.sqrt(np.sum(x**2, axis=1))[:, None]

    N_bodies = len(masses)
    acceleration = np.zeros((N_bodies, 3))
    for i in range(N_bodies):
        r = positions[i+1:] - positions[i]
        norm_r = norm(r)
        force = - G * r / (norm_r**3)
        acceleration[i] -= np.sum(masses[i+1:, None] * force, axis=0)
        acceleration[i+1:] += force * masses[i]
    return acceleration


def leapfrog_integrator(
    positions_init: np.ndarray,
    velocities_init: np.ndarray,
    masses: np.ndarray,
    dt: float,  # use 0.8 days
    N_steps: int,  # 300 years / 0.8 days
) -> tuple[np.ndarray, np.ndarray]:
    """
    The leapfrog integrator.

    Parameters
    ----------
    positions_init : ndarray
        Initial positions, shape (N_bodies, 3).
    velocities_init : ndarray
        Initial velocities, shape (N_bodies, 3).
    masses : ndarray
        Masses of the bodies, shape (N_bodies,).
    dt : float
        Time step in years.
    N_steps : int
        Number of integration steps.

    Returns
    -------
    positions : ndarray
        Positions at all time steps, shape (N_steps + 1, N_bodies, 3).
    velocities : ndarray
        Velocities at all time steps, shape (N_steps + 1, N_bodies, 3).
    """
    positions = np.zeros((N_steps + 1, len(masses), 3))
    velocities = np.zeros((N_steps + 1, len(masses), 3))

    positions[0] = positions_init
    velocities[0] = velocities_init + 0.5 * dt * compute_accelerations(positions_init, masses=masses)
    for step in tqdm(range(N_steps)):
        positions[step + 1] = positions[step] + dt * velocities[step]
        velocities[step + 1] = velocities[step] + dt * compute_accelerations(positions[step + 1], masses=masses)

    return positions, np.zeros(
        (N_steps + 1, len(masses), 3)
    )


def another_integrator(
    positions_init: np.ndarray,
    velocities_init: np.ndarray,
    masses: np.ndarray,
    dt: float,
    N_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the trajectories using another integration method.

    Parameters
    ----------
    positions_init : ndarray
        Initial positions, shape (N_bodies, 3)
    velocities_init : ndarray
        Initial velocities, shape (N_bodies, 3)
    masses : ndarray
        Masses of the bodies, shape (N_bodies,)
    dt : float
        Time step in years
    N_steps : int
        Number of integration steps

    Returns
    -------
    positions : ndarray
        Positions at all time steps, shape (N_steps + 1, N_bodies, 3)
    velocities : ndarray
        Velocities at all time steps, shape (N_steps + 1, N_bodies, 3)
    """
    def f(pos_vel):
        return np.array([pos_vel[1], compute_accelerations(pos_vel[0], masses=masses)])
    positions = np.zeros((N_steps + 1, len(masses), 3))
    velocities = np.zeros((N_steps + 1, len(masses), 3))
    positions[0] = positions_init
    velocities[0] = velocities_init
    current_position_velocity = np.array([positions_init, velocities_init])
    for step in tqdm(range(N_steps)):
        k_1 = dt*f(current_position_velocity)
        k_2 = dt*f(current_position_velocity+.5*k_1)
        k_3 = dt*f(current_position_velocity+.5*k_2)
        k_4 = dt*f(current_position_velocity+k_3)
        current_position_velocity += (k_1+2*k_2+2*k_3+k_4)/6
        positions[step+1], velocities[step+1] = current_position_velocity
    return positions, velocities


##### Plots #####


def plot_initial_positions(
    positions: np.ndarray,
    body_names: list[str],
    output_dir: str,
    filename: str,
) -> None:
    """
    Plot initial positions in the (x,y) and (x,z) planes.

    Parameters
    ----------
    positions : ndarray
        Initial positions, shape (N_bodies, 3)
    body_names : list of str
        Names of the bodies
    output_dir : str
        Directory to save the plots
    """

    x, y, z = positions.T
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for i, obj in enumerate(body_names):
        ax[0].scatter(x[i], y[i], label=obj)
        ax[1].scatter(x[i], z[i], label=obj)
    ax[0].set_aspect("equal", "box")
    ax[1].set_aspect("equal", "box")
    ax[0].set(xlabel="X [AU]", ylabel="Y [AU]")
    ax[1].set(xlabel="X [AU]", ylabel="Z [AU]")
    plt.legend(loc=(1.05, 0))
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close(fig)


def plot_orbits_xy(
    positions: np.ndarray,
    body_names: list[str],
    output_dir: str,
    filename: str,
) -> None:
    """
    Plot the orbits of all bodies in the x-y plane.

    Parameters
    ----------
    positions : ndarray
        Positions at all time steps, shape (N_steps, N_bodies, 3)
    body_names : list of str
        Names of the bodies
    output_dir : str
        Directory where plot is saved
    """

    # For visibility, you may want to do two versions of this plot:
    # one with all planets, and another zoomed in on the four inner planets
    x, y, z = positions.T
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for i, obj in enumerate(body_names):
        ax[0].plot(x[i, :], y[i, :], label=obj)
        ax[1].plot(x[i, :], y[i, :], label=obj)
    ax[0].set_aspect("equal", "box")
    ax[0].set(xlabel="X [AU]", ylabel="Y [AU]")
    ax[1].set_aspect("equal", "box")
    ax[1].set(xlabel="X [AU]", ylabel="Y [AU]")
    ax[1].set_xlim(-2, 2)
    ax[1].set_ylim(-2, 2)
    plt.legend(loc=(1.05, 0))
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close(fig)


def z_vs_time(
    times: np.ndarray,
    positions: np.ndarray,
    body_names: list[str],
    output_dir: str,
    filename: str,
) -> None:
    """
    Plot z position as a function of time.

    Parameters
    ----------
    times : ndarray
        Times, shape (N_steps,)
    positions : ndarray
        Positions, shape (N_steps, N_bodies, 3)
    body_names : list of str
        Names of the bodies
    output_dir : str
        Directory where to save the plot
    filename : str
        Output filename
    """

    x, y, z = positions.T
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for i, obj in enumerate(body_names):
        ax.plot(times, z[i, :], label=obj)
    ax.set(
        xlabel="Time [yr]",
        ylabel="z [AU]",
        title="z position as a function of time",
    )
    ax.legend(fontsize=8)
    plt.legend(loc=(1.05, 0))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close(fig)


def plot_x_difference_vs_time(
    times: np.ndarray,
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    body_names: list[str],
    output_dir: str,
    filename: str,
) -> None:
    """
    Plot difference in x positions between two integration methods.

    Parameters
    ----------
    times : ndarray
        Times, shape (N_steps,)
    positions_a : ndarray
        Positions from first method, shape (N_steps, N_bodies, 3)
    positions_b : ndarray
        Positions from second method, shape (N_steps, N_bodies, 3)
    body_names : list of str
        Names of the bodies
    output_dir : str
        Directory where plot is saved
    filename : str
        Output filename
    """

    x_a, y_a, z_a = positions_a.T
    x_b, y_b, z_b = positions_b.T
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for i, obj in enumerate(np.flip(body_names)):
        ax.plot(times, np.abs((x_b[i, :] - x_a[i, :])/x_a[i, :]), label=obj)
    ax.set(xlabel="Time [yr]", ylabel="Relative Difference [AU]")
    ax.set_yscale('log')
    plt.legend(loc=(1.05, 0))
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close(fig)


def make_movie_with_matplotlib(
    positions: np.ndarray,
    times: np.ndarray,
    body_names: list[str],
    output_dir: str,
    frame_interval: int,
    movie_name: str = "solar_system_movie.mp4",
    fps: int = 30,
) -> str:
    """
    Create a Solar System animation directly with Matplotlib.

    Parameters
    ----------
    positions : ndarray
        Positions, shape (N_steps, N_bodies, 3)
    body_names : list of str
        Names of the bodies
    output_dir : str
        Directory where the movie is saved
    frame_interval : int
        Number of simulation steps between animation frames
    movie_name : str, optional
        Name of the output movie file
    fps : int, optional
        Frames per second of the output movie

    Returns
    -------
    movie_path : str
        Path to the created movie
    """
    os.makedirs(output_dir, exist_ok=True)
    movie_path = os.path.join(output_dir, movie_name)

    positions_plot = positions.copy()  # shape (N_steps, N_bodies, 3)
    positions_plot = (
        positions_plot - positions[:, 0, :][:, None, :]
    )  # shift relative to the Sun

    frame_steps = list(range(0, positions.shape[0], frame_interval))

    fig, ax = plt.subplots()

    scatters = []
    for name in body_names:
        scatter = ax.scatter([], [], label=name)
        scatters.append(scatter)

    ax.set(
        xlabel="x [AU]",
        ylabel="y [AU]",
        title="Solar System movie",
        aspect="equal",
        xlim=(-35, 35),
        ylim=(-35, 35),
    )
    ax.legend(fontsize=8)
    plt.tight_layout()

    def init():
        for scatter in scatters:
            scatter.set_offsets(np.empty((0, 2)))
        return scatters

    def update(frame_idx):
        step = frame_steps[frame_idx]

        for i, scatter in enumerate(scatters):
            x = positions_plot[step, i, 0]
            y = positions_plot[step, i, 1]
            scatter.set_offsets(np.array([[x, y]]))

        ax.set_title(f"Solar System, +{int(times[step])} yrs")
        return scatters

    ani = FuncAnimation(
        fig,
        update,
        frames=len(frame_steps),
        init_func=init,
        blit=False,
    )

    writer = animation.FFMpegWriter(fps=fps)
    ani.save(movie_path, writer=writer, dpi=150)

    plt.close(fig)

    return movie_path


def main() -> None:
    output_dir = "Plots"
    os.makedirs(output_dir, exist_ok=True)

    body_names = list(bodies_with_masses.keys())
    masses = np.array(list(bodies_with_masses.values()))

    # current time
    time = Time.now()

    positions_init, velocities_init = get_initial_conditions(time=time)

    # (a)
    plot_initial_positions(
        positions=positions_init,
        body_names=body_names,
        output_dir=output_dir,
        filename="initial_positions.png",
    )

    # Parameters for the exercise
    total_time = 300.0  # years
    dt = 0.8 / 365.25  # 0.8 days in years
    N_steps = int(total_time / dt)

    times = np.arange(N_steps + 1) * dt

    # (b)
    positions_lf, velocities_lf = leapfrog_integrator(
        positions_init=positions_init,
        velocities_init=velocities_init,
        masses=masses,
        dt=dt,
        N_steps=N_steps,
    )

    plot_orbits_xy(
        positions=positions_lf,
        body_names=body_names,
        output_dir=output_dir,
        filename="orbits_xy_leapfrog.png",
    )

    z_vs_time(
        times=times,
        positions=positions_lf,
        body_names=body_names,
        output_dir=output_dir,
        filename="z_vs_time_leapfrog.png",
    )
    # (c)
    positions_another, velocities_another = another_integrator(
        positions_init=positions_init,
        velocities_init=velocities_init,
        masses=masses,
        dt=dt,
        N_steps=N_steps,
    )

    plot_orbits_xy(
        positions=positions_another,
        body_names=body_names,
        output_dir=output_dir,
        filename="orbits_xy_another_method.png",
    )

    z_vs_time(
        times=times,
        positions=positions_another,
        body_names=body_names,
        output_dir=output_dir,
        filename="z_vs_time_another_method.png",
    )

    plot_x_difference_vs_time(
        times=times,
        positions_a=positions_lf,
        positions_b=positions_another,
        body_names=body_names,
        output_dir=output_dir,
        filename="x_difference_another_method_minus_leapfrog.png",
    )

    # (d): optional
    movie_path = make_movie_with_matplotlib(
        positions=positions_lf,
        times=times,
        body_names=body_names,
        output_dir="Plots",
        frame_interval=200,
        movie_name="solar_system_movie.mp4",
        fps=30,
    )


if __name__ == "__main__":
    main()
