"""
Experiment 003d: Resolution-Normalized Metrics

Purpose
-------
Experiment 003c showed that raw L2 separation between two kinetic
representations increased as the phase-space histogram became finer.

That result is ambiguous because the numerical value of an L2 norm
depends on histogram resolution.

This experiment repeats the same hidden-microscopic-state construction
but evaluates the resulting kinetic separation using probability-based
metrics:

    Total variation distance
    Hellinger distance

These metrics provide a more resolution-aware measure of how
distinguishable the two kinetic states are.

The underlying microscopic model is the same HMF-style particle system
used in Experiments 003b and 003c.

This is a controlled computational experiment, not a claim about the
physical Boltzmann equation.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Allow direct execution from the repository root.
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

N_PARTICLES = 1024

DOMAIN_LENGTH = 2.0 * np.pi

DT = 0.01
T_FINAL = 20.0

RESOLUTIONS = [8, 16, 32, 64]

V_MIN = -2.5
V_MAX = 2.5

RNG_SEED = 12345


# ---------------------------------------------------------------------
# Initial microscopic state
# ---------------------------------------------------------------------

def initial_state(n_particles):
    """
    Construct the baseline microscopic state used in 003b/003c.
    """
    rng = np.random.default_rng(RNG_SEED)

    u = rng.random(n_particles)

    x = (
        DOMAIN_LENGTH * u
        + 0.35 * np.sin(2.0 * np.pi * u)
    ) % DOMAIN_LENGTH

    v = (
        0.35 * np.cos(x)
        + 0.12 * rng.normal(size=n_particles)
    )

    return x, v


# ---------------------------------------------------------------------
# Hidden-state construction
# ---------------------------------------------------------------------

def hidden_state(
    x,
    v,
    resolution,
    x_fraction=0.37,
    v_fraction=0.23,
):
    """
    Construct a second microscopic state by moving every particle
    within its phase-space cell.

    The displacement is chosen so that the particle remains inside
    the same histogram cell.

    Therefore the two states have exactly the same histogram at the
    chosen resolution at t=0.
    """
    x_edges = np.linspace(
        0.0,
        DOMAIN_LENGTH,
        resolution + 1,
    )

    v_edges = np.linspace(
        V_MIN,
        V_MAX,
        resolution + 1,
    )

    ix = np.searchsorted(x_edges, x, side="right") - 1
    iv = np.searchsorted(v_edges, v, side="right") - 1

    ix = np.clip(ix, 0, resolution - 1)
    iv = np.clip(iv, 0, resolution - 1)

    dx = x_edges[1] - x_edges[0]
    dv = v_edges[1] - v_edges[0]

    x_new = (
        x
        + x_fraction * dx
    ) % DOMAIN_LENGTH

    v_new = v + v_fraction * dv

    # Keep velocity inside the histogram range.
    v_new = np.clip(
        v_new,
        V_MIN + 1e-12,
        V_MAX - 1e-12,
    )

    # The construction above normally stays inside the original cell.
    # The checks below make that assumption explicit.
    ix_new = np.searchsorted(
        x_edges,
        x_new,
        side="right",
    ) - 1

    iv_new = np.searchsorted(
        v_edges,
        v_new,
        side="right",
    ) - 1

    ix_new = np.clip(ix_new, 0, resolution - 1)
    iv_new = np.clip(iv_new, 0, resolution - 1)

    if not np.array_equal(ix, ix_new):
        raise RuntimeError(
            "Hidden x displacement crossed a histogram cell."
        )

    if not np.array_equal(iv, iv_new):
        raise RuntimeError(
            "Hidden v displacement crossed a histogram cell."
        )

    return x_new, v_new


# ---------------------------------------------------------------------
# Microscopic dynamics
# ---------------------------------------------------------------------

def acceleration(x):
    """
    HMF-style mean-field acceleration.

    F(x) = -Mx sin(x) + My cos(x)
    """
    mx = np.mean(np.cos(x))
    my = np.mean(np.sin(x))

    return (
        -mx * np.sin(x)
        + my * np.cos(x)
    )


def velocity_verlet_step(x, v, dt):
    """
    One velocity-Verlet step.
    """
    a = acceleration(x)

    v_half = v + 0.5 * dt * a

    x_new = (
        x + dt * v_half
    ) % DOMAIN_LENGTH

    a_new = acceleration(x_new)

    v_new = (
        v_half
        + 0.5 * dt * a_new
    )

    return x_new, v_new


# ---------------------------------------------------------------------
# Kinetic representation
# ---------------------------------------------------------------------

def phase_space_probability(x, v, resolution):
    """
    Return normalized phase-space bin probabilities.

    The sum of all bins is exactly one, apart from floating-point
    roundoff.

    Unlike a density-based L2 norm, these probabilities provide a
    resolution-aware representation for total variation and
    Hellinger distances.
    """
    x_edges = np.linspace(
        0.0,
        DOMAIN_LENGTH,
        resolution + 1,
    )

    v_edges = np.linspace(
        V_MIN,
        V_MAX,
        resolution + 1,
    )

    counts, _, _ = np.histogram2d(
        x,
        v,
        bins=[
            x_edges,
            v_edges,
        ],
    )

    probabilities = counts / np.sum(counts)

    return probabilities


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def total_variation(prob_a, prob_b):
    """
    Total variation distance between two discrete probability
    distributions.
    """
    return 0.5 * np.sum(
        np.abs(prob_a - prob_b)
    )


def hellinger_distance(prob_a, prob_b):
    """
    Hellinger distance between two discrete probability
    distributions.
    """
    sqrt_a = np.sqrt(prob_a)
    sqrt_b = np.sqrt(prob_b)

    return (
        np.sqrt(
            0.5
            * np.sum(
                (sqrt_a - sqrt_b) ** 2
            )
        )
    )


def density_l2(prob_a, prob_b, resolution):
    """
    Convert probability differences to density differences and return
    the conventional L2 norm.

    This is retained only as a diagnostic comparison with 003c.
    """
    dx = DOMAIN_LENGTH / resolution
    dv = (V_MAX - V_MIN) / resolution

    density_difference = (
        (prob_a - prob_b)
        / (dx * dv)
    )

    return np.sqrt(
        np.sum(
            density_difference ** 2
        ) * dx * dv
    )


# ---------------------------------------------------------------------
# One resolution
# ---------------------------------------------------------------------

def run_resolution(resolution):
    """
    Run one A/B hidden-state comparison at a specified resolution.
    """
    x_a, v_a = initial_state(N_PARTICLES)

    x_b, v_b = hidden_state(
        x_a,
        v_a,
        resolution,
    )

    initial_prob_a = phase_space_probability(
        x_a,
        v_a,
        resolution,
    )

    initial_prob_b = phase_space_probability(
        x_b,
        v_b,
        resolution,
    )

    initial_tv = total_variation(
        initial_prob_a,
        initial_prob_b,
    )

    initial_h = hellinger_distance(
        initial_prob_a,
        initial_prob_b,
    )

    if not np.isclose(
        initial_tv,
        0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            "Initial kinetic states are not identical."
        )

    times = []
    tv_history = []
    hellinger_history = []
    l2_history = []

    n_steps = int(
        round(T_FINAL / DT)
    )

    sample_interval = max(
        1,
        n_steps // 500,
    )

    for step in range(n_steps + 1):

        if (
            step % sample_interval == 0
            or step == n_steps
        ):
            prob_a = phase_space_probability(
                x_a,
                v_a,
                resolution,
            )

            prob_b = phase_space_probability(
                x_b,
                v_b,
                resolution,
            )

            times.append(step * DT)

            tv_history.append(
                total_variation(
                    prob_a,
                    prob_b,
                )
            )

            hellinger_history.append(
                hellinger_distance(
                    prob_a,
                    prob_b,
                )
            )

            l2_history.append(
                density_l2(
                    prob_a,
                    prob_b,
                    resolution,
                )
            )

        if step < n_steps:
            x_a, v_a = velocity_verlet_step(
                x_a,
                v_a,
                DT,
            )

            x_b, v_b = velocity_verlet_step(
                x_b,
                v_b,
                DT,
            )

    return {
        "resolution": resolution,
        "initial_tv": initial_tv,
        "initial_hellinger": initial_h,
        "times": np.asarray(times),
        "tv": np.asarray(tv_history),
        "hellinger": np.asarray(hellinger_history),
        "l2": np.asarray(l2_history),
    }


# ---------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------

def run_experiment():
    results = []

    for resolution in RESOLUTIONS:
        print(
            f"Running resolution {resolution}x{resolution}..."
        )

        result = run_resolution(
            resolution
        )

        results.append(result)

    return results


# ---------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------

def save_outputs(results):
    results_dir = ROOT / "results"
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # CSV
    # ---------------------------------------------------------------

    csv_path = (
        results_dir
        / "experiment_003d_summary.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "resolution,"
            "maximum_tv,"
            "final_tv,"
            "maximum_hellinger,"
            "final_hellinger,"
            "maximum_l2,"
            "final_l2\n"
        )

        for result in results:
            handle.write(
                f"{result['resolution']},"
                f"{np.max(result['tv']):.12e},"
                f"{result['tv'][-1]:.12e},"
                f"{np.max(result['hellinger']):.12e},"
                f"{result['hellinger'][-1]:.12e},"
                f"{np.max(result['l2']):.12e},"
                f"{result['l2'][-1]:.12e}\n"
            )

    # ---------------------------------------------------------------
    # Summary text
    # ---------------------------------------------------------------

    summary_path = (
        results_dir
        / "experiment_003d_summary.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "Experiment 003d: "
            "Resolution-Normalized Metrics\n"
        )
        handle.write(
            "========================================\n\n"
        )

        handle.write(
            f"Particles: {N_PARTICLES}\n"
        )
        handle.write(
            f"Time step: {DT}\n"
        )
        handle.write(
            f"Final time: {T_FINAL}\n"
        )
        handle.write(
            "Resolutions: "
            + ", ".join(
                f"{r}x{r}"
                for r in RESOLUTIONS
            )
            + "\n\n"
        )

        handle.write(
            "Initial-state condition\n"
        )
        handle.write(
            "------------------------\n"
        )
        handle.write(
            "The microscopic states differ,\n"
        )
        handle.write(
            "but their kinetic histograms are "
            "identical at each tested resolution.\n\n"
        )

        handle.write(
            "Resolution results\n"
        )
        handle.write(
            "------------------\n"
        )

        handle.write(
            "resolution,"
            "maximum_TV,"
            "final_TV,"
            "maximum_Hellinger,"
            "final_Hellinger,"
            "maximum_L2,"
            "final_L2\n"
        )

        for result in results:
            handle.write(
                f"{result['resolution']},"
                f"{np.max(result['tv']):.12e},"
                f"{result['tv'][-1]:.12e},"
                f"{np.max(result['hellinger']):.12e},"
                f"{result['hellinger'][-1]:.12e},"
                f"{np.max(result['l2']):.12e},"
                f"{result['l2'][-1]:.12e}\n"
            )

    # ---------------------------------------------------------------
    # Plot 1: Total variation
    # ---------------------------------------------------------------

    plt.figure()

    for result in results:
        plt.plot(
            result["times"],
            result["tv"],
            label=f"{result['resolution']}x"
                  f"{result['resolution']}",
        )

    plt.xlabel("Time")
    plt.ylabel("Total variation distance")
    plt.title(
        "003d: Kinetic-state separation "
        "(total variation)"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_003d_total_variation.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Plot 2: Hellinger distance
    # ---------------------------------------------------------------

    plt.figure()

    for result in results:
        plt.plot(
            result["times"],
            result["hellinger"],
            label=f"{result['resolution']}x"
                  f"{result['resolution']}",
        )

    plt.xlabel("Time")
    plt.ylabel("Hellinger distance")
    plt.title(
        "003d: Kinetic-state separation "
        "(Hellinger)"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_003d_hellinger.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Plot 3: Compare metrics at final time
    # ---------------------------------------------------------------

    resolutions = [
        result["resolution"]
        for result in results
    ]

    final_tv = [
        result["tv"][-1]
        for result in results
    ]

    final_h = [
        result["hellinger"][-1]
        for result in results
    ]

    final_l2 = [
        result["l2"][-1]
        for result in results
    ]

    plt.figure()

    plt.plot(
        resolutions,
        final_tv,
        marker="o",
        label="Total variation",
    )

    plt.plot(
        resolutions,
        final_h,
        marker="o",
        label="Hellinger",
    )

    plt.plot(
        resolutions,
        final_l2,
        marker="o",
        label="Density L2",
    )

    plt.xlabel("Phase-space resolution per dimension")
    plt.ylabel("Final separation")
    plt.title(
        "003d: Resolution dependence of "
        "different separation metrics"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_003d_metric_comparison.png",
        dpi=150,
    )

    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    results = run_experiment()

    save_outputs(results)

    print()
    print(
        "Experiment 003d: "
        "Resolution-Normalized Metrics"
    )
    print(
        "========================================"
    )
    print()

    print(
        "Resolution results"
    )
    print(
        "------------------"
    )

    for result in results:

        print(
            f"resolution={result['resolution']:2d}, "
            f"maximum_TV="
            f"{np.max(result['tv']):.12e}, "
            f"final_TV="
            f"{result['tv'][-1]:.12e}, "
            f"maximum_Hellinger="
            f"{np.max(result['hellinger']):.12e}, "
            f"final_Hellinger="
            f"{result['hellinger'][-1]:.12e}, "
            f"maximum_L2="
            f"{np.max(result['l2']):.12e}, "
            f"final_L2="
            f"{result['l2'][-1]:.12e}"
        )

    print()
    print(
        "Outputs written to results/"
    )
