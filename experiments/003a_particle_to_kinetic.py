"""
Experiment 003a: Particle -> Kinetic Representation.

This experiment establishes the first stage of the particle-to-kinetic
description.

Microscopic state:

    X = (x_1, v_1, ..., x_N, v_N)

Kinetic representation:

    f(x, v, t)

The particle ensemble is represented by a normalized phase-space
histogram.

This experiment deliberately uses non-interacting particles:

    dx_i/dt = v_i
    dv_i/dt = 0

The purpose is to validate the representation and its moments before
introducing interactions and hidden-state dependence in subsequent
experiments.

This is a methodological experiment, not a simulation of a realistic
fluid.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Repository setup
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_PARTICLES = 1024

DOMAIN_LENGTH = 1.0

DT = 0.002
T_FINAL = 4.0

N_X_BINS = 32
N_V_BINS = 32

V_MIN = -2.5
V_MAX = 2.5

RNG_SEED = 12345


# ---------------------------------------------------------------------------
# Particle system
# ---------------------------------------------------------------------------

def initial_particles(n, seed):
    """
    Construct an initial particle ensemble.

    Positions are uniformly distributed.

    Velocities contain a spatially coherent component plus thermal-like
    spread. The velocity distribution is deliberately simple; later
    experiments can replace this with more physically motivated
    distributions.
    """

    rng = np.random.default_rng(seed)

    x = rng.uniform(
        0.0,
        DOMAIN_LENGTH,
        size=n,
    )

    coherent_velocity = np.sin(
        2.0 * np.pi * x / DOMAIN_LENGTH
    )

    thermal_velocity = rng.normal(
        loc=0.0,
        scale=0.25,
        size=n,
    )

    v = coherent_velocity + thermal_velocity

    return x, v


def particle_step(x, v, dt):
    """
    Advance non-interacting particles.

    Periodic spatial boundary conditions are applied.
    """

    x_next = (
        x + dt * v
    ) % DOMAIN_LENGTH

    v_next = v.copy()

    return x_next, v_next


# ---------------------------------------------------------------------------
# Kinetic representation
# ---------------------------------------------------------------------------

def phase_space_histogram(
    x,
    v,
    n_x_bins,
    n_v_bins,
):
    """
    Construct a normalized phase-space density f(x,v).

    The returned array satisfies approximately

        sum(f * dx * dv) = 1.
    """

    histogram, x_edges, v_edges = np.histogram2d(
        x,
        v,
        bins=[
            n_x_bins,
            n_v_bins,
        ],
        range=[
            [0.0, DOMAIN_LENGTH],
            [V_MIN, V_MAX],
        ],
    )

    dx = x_edges[1] - x_edges[0]
    dv = v_edges[1] - v_edges[0]

    f = histogram / (
        len(x) * dx * dv
    )

    return (
        f,
        x_edges,
        v_edges,
    )


# ---------------------------------------------------------------------------
# Kinetic moments
# ---------------------------------------------------------------------------

def kinetic_moments(
    f,
    x_edges,
    v_edges,
):
    """
    Calculate spatial density, mean velocity, and velocity variance.
    """

    dx = x_edges[1] - x_edges[0]
    dv = v_edges[1] - v_edges[0]

    x_centers = (
        0.5
        * (
            x_edges[:-1]
            + x_edges[1:]
        )
    )

    v_centers = (
        0.5
        * (
            v_edges[:-1]
            + v_edges[1:]
        )
    )

    # Integrate over velocity.
    density = np.sum(
        f,
        axis=1,
    ) * dv

    momentum = np.sum(
        f
        * v_centers[np.newaxis, :],
        axis=1,
    ) * dv

    mean_velocity = np.zeros_like(
        density
    )

    valid = density > 0.0

    mean_velocity[valid] = (
        momentum[valid]
        / density[valid]
    )

    # Velocity variance.
    variance = np.zeros_like(
        density
    )

    for i in range(len(x_centers)):

        if density[i] > 0.0:

            deviations = (
                v_centers
                - mean_velocity[i]
            )

            variance[i] = (
                np.sum(
                    f[i, :]
                    * deviations**2
                )
                * dv
                / density[i]
            )

    return (
        x_centers,
        density,
        mean_velocity,
        variance,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def normalization(f, x_edges, v_edges):
    """Calculate integral of f over phase space."""

    dx = x_edges[1] - x_edges[0]
    dv = v_edges[1] - v_edges[0]

    return np.sum(f) * dx * dv


def run_experiment():
    """Run the particle-to-kinetic representation experiment."""

    x, v = initial_particles(
        N_PARTICLES,
        RNG_SEED,
    )

    n_steps = int(
        round(T_FINAL / DT)
    )

    snapshots = []

    for step in range(n_steps + 1):

        if step in [
            0,
            n_steps // 2,
            n_steps,
        ]:

            f, x_edges, v_edges = (
                phase_space_histogram(
                    x,
                    v,
                    N_X_BINS,
                    N_V_BINS,
                )
            )

            moments = kinetic_moments(
                f,
                x_edges,
                v_edges,
            )

            snapshots.append(
                {
                    "time": step * DT,
                    "f": f.copy(),
                    "x_edges": x_edges.copy(),
                    "v_edges": v_edges.copy(),
                    "moments": moments,
                    "normalization":
                        normalization(
                            f,
                            x_edges,
                            v_edges,
                        ),
                }
            )

        if step == n_steps:
            break

        x, v = particle_step(
            x,
            v,
            DT,
        )

    return snapshots


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_outputs(snapshots):
    """Save plots and numerical diagnostics."""

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------------
    # Phase-space snapshots
    # ---------------------------------------------------------------

    for snapshot in snapshots:

        time = snapshot["time"]
        f = snapshot["f"]
        x_edges = snapshot["x_edges"]
        v_edges = snapshot["v_edges"]

        plt.figure(
            figsize=(8, 6)
        )

        plt.imshow(
            f.T,
            origin="lower",
            aspect="auto",
            extent=[
                x_edges[0],
                x_edges[-1],
                v_edges[0],
                v_edges[-1],
            ],
        )

        plt.xlabel("Position x")
        plt.ylabel("Velocity v")
        plt.title(
            f"Phase-space density at t={time:.2f}"
        )
        plt.colorbar(
            label="f(x,v)"
        )
        plt.tight_layout()

        filename = (
            results_dir
            / (
                "experiment_003a_phase_space_"
                f"{time:.2f}.png"
            )
        )

        plt.savefig(
            filename,
            dpi=150,
        )

        plt.close()

    # ---------------------------------------------------------------
    # Final spatial moments
    # ---------------------------------------------------------------

    final = snapshots[-1]

    (
        x_centers,
        density,
        mean_velocity,
        variance,
    ) = final["moments"]

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        x_centers,
        density,
        label="Density",
    )

    plt.plot(
        x_centers,
        mean_velocity,
        label="Mean velocity",
    )

    plt.plot(
        x_centers,
        variance,
        label="Velocity variance",
    )

    plt.xlabel("Position x")
    plt.ylabel("Moment value")
    plt.title(
        "Experiment 003a: Kinetic Moments"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_003a_moments.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    summary_path = (
        results_dir
        / "experiment_003a_summary.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Experiment 003a: "
            "Particle -> Kinetic Representation\n"
        )

        f.write(
            "=====================================\n\n"
        )

        f.write(
            f"Particles: {N_PARTICLES}\n"
        )

        f.write(
            f"Time step: {DT}\n"
        )

        f.write(
            f"Final time: {T_FINAL}\n"
        )

        f.write(
            f"Spatial bins: {N_X_BINS}\n"
        )

        f.write(
            f"Velocity bins: {N_V_BINS}\n\n"
        )

        f.write(
            "Phase-space normalization\n"
            "--------------------------\n"
        )

        for snapshot in snapshots:

            f.write(
                f"t={snapshot['time']:.3f}, "
                f"integral(f dx dv)="
                f"{snapshot['normalization']:.12e}\n"
            )

        f.write("\n")

        final_moments = snapshots[-1][
            "moments"
        ]

        (
            _,
            density,
            mean_velocity,
            variance,
        ) = final_moments

        f.write(
            "Final-state diagnostics\n"
            "-----------------------\n"
        )

        f.write(
            f"Mean density = "
            f"{np.mean(density):.12e}\n"
        )

        f.write(
            f"Mean velocity = "
            f"{np.mean(mean_velocity):.12e}\n"
        )

        f.write(
            f"Mean velocity variance = "
            f"{np.mean(variance):.12e}\n"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    snapshots = run_experiment()

    save_outputs(snapshots)

    print()
    print(
        "Experiment 003a complete."
    )
    print(
        "Results written to: results/"
    )
