"""
Experiment 003c: Resolution Dependence of Hidden-State Influence.

Question:

    Does hidden-state dependence decrease systematically as the
    kinetic representation becomes more detailed?

We construct two microscopic states:

    X_A(0) != X_B(0)

while making them identical under the finest kinetic representation:

    C_64[X_A(0)] = C_64[X_B(0)]

Because the hidden perturbation remains inside each 64x64 phase-space
cell, the two states are also identical under the coarser
representations:

    C_8[X_A(0)]  = C_8[X_B(0)]
    C_16[X_A(0)] = C_16[X_B(0)]
    C_32[X_A(0)] = C_32[X_B(0)]
    C_64[X_A(0)] = C_64[X_B(0)]

The same microscopic trajectories are then observed at all four
resolutions.

This isolates the effect of representational resolution.

The central quantity is:

    S_R(t) = ||C_R[X_A(t)] - C_R[X_B(t)]||

where R is the phase-space resolution.

This remains a controlled particle-model experiment. It is not yet a
test of the Boltzmann equation or of Navier-Stokes.
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

DOMAIN_LENGTH = 2.0 * np.pi

DT = 0.01
T_FINAL = 20.0

RESOLUTIONS = [
    8,
    16,
    32,
    64,
]

V_MIN = -2.5
V_MAX = 2.5

RNG_SEED = 12345

# Hidden perturbation is constructed inside 64x64 cells.
FINE_RESOLUTION = 64

HIDDEN_SHIFT_X = 0.37
HIDDEN_SHIFT_V = 0.23

SNAPSHOT_INTERVAL = 10


# ---------------------------------------------------------------------------
# Initial microscopic state
# ---------------------------------------------------------------------------

def initial_particles(n, seed):
    """
    Construct the reference microscopic particle ensemble.
    """

    rng = np.random.default_rng(seed)

    u = (
        np.arange(n) + 0.5
    ) / n

    x = (
        DOMAIN_LENGTH * u
        + 0.35
        * np.sin(
            2.0 * np.pi * u
        )
    ) % DOMAIN_LENGTH

    coherent_velocity = (
        0.35
        * np.cos(x)
    )

    thermal_velocity = (
        0.12
        * rng.standard_normal(n)
    )

    v = (
        coherent_velocity
        + thermal_velocity
    )

    return x, v


# ---------------------------------------------------------------------------
# Hidden-state construction
# ---------------------------------------------------------------------------

def shifted_within_bins(
    values,
    edges,
    shift,
):
    """
    Move each value to another point inside its existing bin.

    The bin membership is preserved exactly.
    """

    indices = (
        np.searchsorted(
            edges,
            values,
            side="right",
        )
        - 1
    )

    indices = np.clip(
        indices,
        0,
        len(edges) - 2,
    )

    lower = edges[indices]
    upper = edges[indices + 1]

    width = upper - lower

    fraction = (
        values - lower
    ) / width

    shifted_fraction = (
        fraction + shift
    ) % 1.0

    return (
        lower
        + shifted_fraction * width
    )


def construct_hidden_state(
    x,
    v,
):
    """
    Construct B by moving A within 64x64 phase-space cells.
    """

    x_edges = np.linspace(
        0.0,
        DOMAIN_LENGTH,
        FINE_RESOLUTION + 1,
    )

    v_edges = np.linspace(
        V_MIN,
        V_MAX,
        FINE_RESOLUTION + 1,
    )

    x_hidden = shifted_within_bins(
        x,
        x_edges,
        HIDDEN_SHIFT_X,
    )

    v_hidden = shifted_within_bins(
        v,
        v_edges,
        HIDDEN_SHIFT_V,
    )

    return (
        x_hidden,
        v_hidden,
    )


# ---------------------------------------------------------------------------
# Microscopic dynamics
# ---------------------------------------------------------------------------

def force(x):
    """
    Conservative periodic mean-field force.
    """

    magnetization_x = np.mean(
        np.cos(x)
    )

    magnetization_y = np.mean(
        np.sin(x)
    )

    return (
        -magnetization_x
        * np.sin(x)
        + magnetization_y
        * np.cos(x)
    )


def particle_step(
    x,
    v,
    dt,
):
    """
    Velocity-Verlet integration.
    """

    acceleration_old = force(x)

    v_half = (
        v
        + 0.5
        * dt
        * acceleration_old
    )

    x_new = (
        x
        + dt
        * v_half
    ) % DOMAIN_LENGTH

    acceleration_new = force(
        x_new
    )

    v_new = (
        v_half
        + 0.5
        * dt
        * acceleration_new
    )

    return (
        x_new,
        v_new,
    )


# ---------------------------------------------------------------------------
# Kinetic representation
# ---------------------------------------------------------------------------

def phase_space_histogram(
    x,
    v,
    resolution,
):
    """
    Construct normalized phase-space density at a given resolution.
    """

    histogram, x_edges, v_edges = (
        np.histogram2d(
            x,
            v,
            bins=[
                resolution,
                resolution,
            ],
            range=[
                [
                    0.0,
                    DOMAIN_LENGTH,
                ],
                [
                    V_MIN,
                    V_MAX,
                ],
            ],
        )
    )

    dx = (
        x_edges[1]
        - x_edges[0]
    )

    dv = (
        v_edges[1]
        - v_edges[0]
    )

    f = histogram / (
        len(x)
        * dx
        * dv
    )

    return (
        f,
        dx,
        dv,
        x_edges,
        v_edges,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def l2_separation(
    f_a,
    f_b,
    dx,
    dv,
):
    """
    L2 distance between two kinetic representations.
    """

    return np.sqrt(
        np.sum(
            (f_a - f_b) ** 2
        )
        * dx
        * dv
    )


def rms_separation(
    f_a,
    f_b,
):
    """RMS difference between two kinetic representations."""

    return np.sqrt(
        np.mean(
            (f_a - f_b) ** 2
        )
    )


def initial_representation_errors(
    x_a,
    v_a,
    x_b,
    v_b,
):
    """
    Verify that the initial kinetic states agree at every resolution.
    """

    errors = {}

    for resolution in RESOLUTIONS:

        f_a, dx, dv, _, _ = (
            phase_space_histogram(
                x_a,
                v_a,
                resolution,
            )
        )

        f_b, _, _, _, _ = (
            phase_space_histogram(
                x_b,
                v_b,
                resolution,
            )
        )

        errors[resolution] = (
            l2_separation(
                f_a,
                f_b,
                dx,
                dv,
            )
        )

    return errors


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    """
    Run both microscopic trajectories and observe them at every
    kinetic resolution.
    """

    x_a, v_a = initial_particles(
        N_PARTICLES,
        RNG_SEED,
    )

    x_b, v_b = construct_hidden_state(
        x_a,
        v_a,
    )

    initial_errors = (
        initial_representation_errors(
            x_a,
            v_a,
            x_b,
            v_b,
        )
    )

    initial_microscopic_difference = (
        np.sqrt(
            np.mean(
                (x_a - x_b) ** 2
                + (v_a - v_b) ** 2
            )
        )
    )

    times = []

    l2_history = {
        resolution: []
        for resolution in RESOLUTIONS
    }

    rms_history = {
        resolution: []
        for resolution in RESOLUTIONS
    }

    n_steps = int(
        round(
            T_FINAL / DT
        )
    )

    for step in range(
        n_steps + 1
    ):

        if (
            step % SNAPSHOT_INTERVAL == 0
            or step == n_steps
        ):

            times.append(
                step * DT
            )

            for resolution in RESOLUTIONS:

                (
                    f_a,
                    dx,
                    dv,
                    _,
                    _,
                ) = phase_space_histogram(
                    x_a,
                    v_a,
                    resolution,
                )

                (
                    f_b,
                    _,
                    _,
                    _,
                    _,
                ) = phase_space_histogram(
                    x_b,
                    v_b,
                    resolution,
                )

                l2_history[
                    resolution
                ].append(
                    l2_separation(
                        f_a,
                        f_b,
                        dx,
                        dv,
                    )
                )

                rms_history[
                    resolution
                ].append(
                    rms_separation(
                        f_a,
                        f_b,
                    )
                )

        if step == n_steps:
            break

        x_a, v_a = particle_step(
            x_a,
            v_a,
            DT,
        )

        x_b, v_b = particle_step(
            x_b,
            v_b,
            DT,
        )

    return {
        "times": np.asarray(times),
        "l2_history": {
            r: np.asarray(values)
            for r, values
            in l2_history.items()
        },
        "rms_history": {
            r: np.asarray(values)
            for r, values
            in rms_history.items()
        },
        "initial_errors":
            initial_errors,
        "initial_microscopic_difference":
            initial_microscopic_difference,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_outputs(results):
    """
    Save plots and summary.
    """

    results_dir = ROOT / "results"
    results_dir.mkdir(
        exist_ok=True
    )

    times = results["times"]

    # ---------------------------------------------------------------
    # L2 separation histories
    # ---------------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    for resolution in RESOLUTIONS:

        plt.plot(
            times,
            results["l2_history"][
                resolution
            ],
            label=f"{resolution} x {resolution}",
        )

    plt.xlabel("Time")
    plt.ylabel(
        "L2 kinetic-state separation"
    )

    plt.title(
        "Experiment 003c: Hidden-State Dependence vs Resolution"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_003c_resolution_history.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Maximum separation vs resolution
    # ---------------------------------------------------------------

    maximum_l2 = []

    for resolution in RESOLUTIONS:

        maximum_l2.append(
            np.max(
                results["l2_history"][
                    resolution
                ]
            )
        )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        RESOLUTIONS,
        maximum_l2,
        marker="o",
    )

    plt.xlabel(
        "Phase-space resolution per dimension"
    )

    plt.ylabel(
        "Maximum L2 separation"
    )

    plt.title(
        "Maximum Hidden-State Influence vs Resolution"
    )

    plt.xscale("log", base=2)

    plt.xticks(
        RESOLUTIONS,
        [
            str(r)
            for r in RESOLUTIONS
        ],
    )

    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_003c_max_separation.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    summary_path = (
        results_dir
        / "experiment_003c_summary.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Experiment 003c: "
            "Resolution Dependence of Hidden-State Influence\n"
        )

        f.write(
            "=======================================================\n\n"
        )

        f.write(
            f"Particles: "
            f"{N_PARTICLES}\n"
        )

        f.write(
            f"Time step: "
            f"{DT}\n"
        )

        f.write(
            f"Final time: "
            f"{T_FINAL}\n"
        )

        f.write(
            "Resolutions: "
            + ", ".join(
                f"{r}x{r}"
                for r in RESOLUTIONS
            )
            + "\n\n"
        )

        f.write(
            "Initial microscopic difference\n"
            "-------------------------------\n"
        )

        f.write(
            f"{results['initial_microscopic_difference']:.12e}\n\n"
        )

        f.write(
            "Initial kinetic-state differences\n"
            "----------------------------------\n"
        )

        for resolution in RESOLUTIONS:

            f.write(
                f"{resolution}x{resolution}: "
                f"{results['initial_errors'][resolution]:.12e}\n"
            )

        f.write("\n")

        f.write(
            "Resolution results\n"
            "------------------\n"
        )

        f.write(
            "resolution,maximum_L2,final_L2,"
            "maximum_RMS,final_RMS\n"
        )

        for resolution in RESOLUTIONS:

            l2 = results[
                "l2_history"
            ][resolution]

            rms = results[
                "rms_history"
            ][resolution]

            f.write(
                f"{resolution},"
                f"{np.max(l2):.12e},"
                f"{l2[-1]:.12e},"
                f"{np.max(rms):.12e},"
                f"{rms[-1]:.12e}\n"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    results = run_experiment()

    save_outputs(results)

    print()
    print(
        "Experiment 003c complete."
    )

    print(
        "Results written to: results/"
    )
