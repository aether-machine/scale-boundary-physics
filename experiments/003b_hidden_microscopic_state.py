"""
Experiment 003b: Hidden Microscopic State.

This experiment tests whether two microscopic particle states that have
the same kinetic representation at t=0 can subsequently produce
different kinetic representations.

The defining condition is:

    X_A(0) != X_B(0)

while:

    C[X_A(0)] = C[X_B(0)]

where C is a phase-space histogram.

The microscopic dynamics use a simple periodic mean-field particle
model. This is a controlled kinetic testbed, not a realistic model of
ordinary fluid matter.

The central quantity is the kinetic-state separation:

    S(t) = || C[X_A(t)] - C[X_B(t)] ||

If S(t) becomes nonzero, the chosen kinetic representation is not
dynamically closed with respect to the microscopic dynamics.

This is analogous to Experiment 002d, but moves the test from a
lattice/coarse-grained state to a particle/kinetic representation.
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

N_X_BINS = 32
N_V_BINS = 32

V_MIN = -2.5
V_MAX = 2.5

RNG_SEED = 12345

HIDDEN_SHIFT_X = 0.37
HIDDEN_SHIFT_V = 0.23

SNAPSHOT_INTERVAL = 10


# ---------------------------------------------------------------------------
# Initial microscopic state
# ---------------------------------------------------------------------------

def initial_particles(n, seed):
    """
    Construct a deterministic particle ensemble.

    The spatial distribution contains a coherent density modulation.
    Velocities contain both a coherent component and a small spread.
    """

    rng = np.random.default_rng(seed)

    u = (
        np.arange(n) + 0.5
    ) / n

    x = (
        DOMAIN_LENGTH
        * u
        + 0.35
        * np.sin(
            2.0
            * np.pi
            * u
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
    Move every value to another location inside its existing bin.

    The bin index of every value is preserved exactly.

    This allows us to construct a different microscopic state while
    preserving the phase-space histogram.
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

    shifted = (
        lower
        + shifted_fraction * width
    )

    return shifted


def construct_hidden_state(
    x,
    v,
):
    """
    Construct ensemble B from ensemble A.

    Every particle remains in the same 32x32 phase-space cell, but
    moves to a different position inside that cell.
    """

    x_edges = np.linspace(
        0.0,
        DOMAIN_LENGTH,
        N_X_BINS + 1,
    )

    v_edges = np.linspace(
        V_MIN,
        V_MAX,
        N_V_BINS + 1,
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
    Mean-field periodic force.

    Define:

        M_x = <cos(x)>
        M_y = <sin(x)>

    and:

        F(x) = -M_x sin(x) + M_y cos(x).

    The interaction is conservative and periodic.
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
    Advance the interacting particles using velocity Verlet.
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
):
    """
    Construct normalized f(x,v).
    """

    histogram, x_edges, v_edges = (
        np.histogram2d(
            x,
            v,
            bins=[
                N_X_BINS,
                N_V_BINS,
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
        x_edges,
        v_edges,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def histogram_difference(
    f_a,
    f_b,
    dx,
    dv,
):
    """
    L2 phase-space separation.
    """

    difference = (
        f_a - f_b
    )

    return np.sqrt(
        np.sum(
            difference**2
        )
        * dx
        * dv
    )


def histogram_rms(
    f_a,
    f_b,
):
    """Simple RMS difference between two kinetic states."""

    return np.sqrt(
        np.mean(
            (f_a - f_b) ** 2
        )
    )


def magnetization(x):
    """
    Return magnitude of the collective magnetization.
    """

    mx = np.mean(
        np.cos(x)
    )

    my = np.mean(
        np.sin(x)
    )

    return np.sqrt(
        mx**2 + my**2
    )


def run_experiment():
    """
    Run the two hidden-state trajectories.
    """

    x_a, v_a = initial_particles(
        N_PARTICLES,
        RNG_SEED,
    )

    x_b, v_b = construct_hidden_state(
        x_a,
        v_a,
    )

    # Verify that the initial kinetic representations agree.
    f_a0, x_edges, v_edges = (
        phase_space_histogram(
            x_a,
            v_a,
        )
    )

    f_b0, _, _ = (
        phase_space_histogram(
            x_b,
            v_b,
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

    initial_histogram_difference = (
        histogram_difference(
            f_a0,
            f_b0,
            dx,
            dv,
        )
    )

    initial_microscopic_difference = np.sqrt(
        np.mean(
            (x_a - x_b) ** 2
            + (v_a - v_b) ** 2
        )
    )

    initial_magnetization_difference = abs(
        magnetization(x_a)
        - magnetization(x_b)
    )

    n_steps = int(
        round(
            T_FINAL / DT
        )
    )

    times = []
    separations = []
    rms_separations = []
    magnetization_differences = []

    for step in range(
        n_steps + 1
    ):

        if (
            step % SNAPSHOT_INTERVAL == 0
            or step == n_steps
        ):

            f_a, _, _ = (
                phase_space_histogram(
                    x_a,
                    v_a,
                )
            )

            f_b, _, _ = (
                phase_space_histogram(
                    x_b,
                    v_b,
                )
            )

            separation = (
                histogram_difference(
                    f_a,
                    f_b,
                    dx,
                    dv,
                )
            )

            rms_separation = (
                histogram_rms(
                    f_a,
                    f_b,
                )
            )

            mag_difference = abs(
                magnetization(x_a)
                - magnetization(x_b)
            )

            times.append(
                step * DT
            )

            separations.append(
                separation
            )

            rms_separations.append(
                rms_separation
            )

            magnetization_differences.append(
                mag_difference
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
        "separations": np.asarray(
            separations
        ),
        "rms_separations": np.asarray(
            rms_separations
        ),
        "magnetization_differences":
            np.asarray(
                magnetization_differences
            ),
        "initial_histogram_difference":
            initial_histogram_difference,
        "initial_microscopic_difference":
            initial_microscopic_difference,
        "initial_magnetization_difference":
            initial_magnetization_difference,
        "final_histogram_a":
            f_a,
        "final_histogram_b":
            f_b,
        "x_edges":
            x_edges,
        "v_edges":
            v_edges,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_outputs(results):
    """Save plots and numerical diagnostics."""

    results_dir = ROOT / "results"
    results_dir.mkdir(
        exist_ok=True
    )

    times = results["times"]
    separations = results[
        "separations"
    ]
    rms_separations = results[
        "rms_separations"
    ]
    magnetization_differences = results[
        "magnetization_differences"
    ]

    # ---------------------------------------------------------------
    # Hidden-state separation
    # ---------------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        times,
        separations,
        label="L2 phase-space separation",
    )

    plt.plot(
        times,
        rms_separations,
        label="RMS phase-space separation",
    )

    plt.xlabel("Time")
    plt.ylabel("Separation")
    plt.title(
        "Experiment 003b: Hidden-State Dependence"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_003b_hidden_state_separation.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Magnetization difference
    # ---------------------------------------------------------------

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        times,
        magnetization_differences,
    )

    plt.xlabel("Time")
    plt.ylabel(
        "Difference in magnetization"
    )

    plt.title(
        "Collective Difference Between Hidden States"
    )

    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_003b_magnetization_difference.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Final phase-space difference
    # ---------------------------------------------------------------

    difference = (
        results["final_histogram_a"]
        - results["final_histogram_b"]
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        difference.T,
        origin="lower",
        aspect="auto",
        extent=[
            results["x_edges"][0],
            results["x_edges"][-1],
            results["v_edges"][0],
            results["v_edges"][-1],
        ],
    )

    plt.xlabel("Position x")
    plt.ylabel("Velocity v")

    plt.title(
        "Final Kinetic-State Difference: A - B"
    )

    plt.colorbar(
        label="f_A - f_B"
    )

    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_003b_final_difference.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    summary_path = (
        results_dir
        / "experiment_003b_summary.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Experiment 003b: "
            "Hidden Microscopic State\n"
        )

        f.write(
            "=================================\n\n"
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
            f"Phase-space bins: "
            f"{N_X_BINS} x "
            f"{N_V_BINS}\n\n"
        )

        f.write(
            "Initial-state conditions\n"
            "------------------------\n"
        )

        f.write(
            "X_A(0) != X_B(0)\n"
        )

        f.write(
            "C[X_A(0)] = C[X_B(0)]\n\n"
        )

        f.write(
            "Initial microscopic RMS "
            "difference = "
            f"{results['initial_microscopic_difference']:.12e}\n"
        )

        f.write(
            "Initial kinetic-state L2 "
            "difference = "
            f"{results['initial_histogram_difference']:.12e}\n"
        )

        f.write(
            "Initial magnetization "
            "difference = "
            f"{results['initial_magnetization_difference']:.12e}\n\n"
        )

        f.write(
            "Kinetic-state separation\n"
            "------------------------\n"
        )

        f.write(
            "Maximum L2 separation = "
            f"{np.max(separations):.12e}\n"
        )

        f.write(
            "Final L2 separation = "
            f"{separations[-1]:.12e}\n"
        )

        f.write(
            "Maximum RMS separation = "
            f"{np.max(rms_separations):.12e}\n"
        )

        f.write(
            "Final RMS separation = "
            f"{rms_separations[-1]:.12e}\n"
        )

        f.write(
            "Maximum magnetization "
            "difference = "
            f"{np.max(magnetization_differences):.12e}\n"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    results = run_experiment()

    save_outputs(results)

    print()
    print(
        "Experiment 003b complete."
    )
    print(
        "Results written to: results/"
    )
