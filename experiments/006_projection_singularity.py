"""
Experiment 006: Regular Microdynamics -> Singular Effective Description
=======================================================================

Question
--------
Can a regular, bounded underlying dynamical system produce an apparently
singular effective description when one or more variables are discarded?

This experiment uses a smooth bounded microscopic system and observes only
one variable. We then construct several reduced descriptions of the observed
variable and compare them with the full dynamics.

The experiment is deliberately NOT a model of Navier-Stokes.

It tests a narrower proposition:

    regular underlying dynamics
        ->
    projection / loss of hidden state
        ->
    possible pathology in an effective description

The important distinction is between:

1. the true projected trajectory x(t),
2. the exact non-closed dynamics of x,
3. an approximate autonomous closure dx/dt = F(x).

If the same observed x can correspond to different dx/dt values because of
the hidden variable y, then x alone is not a dynamically closed state.

We also investigate whether a deliberately simple closure can develop a
finite-time blow-up even though the underlying (x,y) system remains bounded.

Outputs
-------
results/006_projection_singularity.csv
results/006_projection_singularity_summary.txt
results/006_micro_and_projection.png
results/006_effective_closure.png
results/006_phase_space.png
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")

DT = 0.002
T_FINAL = 40.0

# Initial microscopic state.
X0 = 0.20
Y0 = 0.80

# Small perturbation used to demonstrate hidden-state dependence.
Y_VARIANTS = np.array([
    -0.80,
    -0.40,
    0.00,
    0.40,
    0.80,
])

# Effective closure sampling.
N_BINS = 80


# ---------------------------------------------------------------------------
# Microscopic system
# ---------------------------------------------------------------------------

def micro_rhs(
    x: float,
    y: float,
) -> tuple[float, float]:
    """
    Smooth bounded microscopic dynamics.

    The system is a nonlinear rotation on the unit disk:

        dx/dt = -y
        dy/dt =  x

    For x(0)^2 + y(0)^2 = 1, the trajectory remains exactly on the
    unit circle.

    Thus the microscopic state remains bounded for all time.

    We deliberately observe only x and discard y.
    """

    dxdt = -y
    dydt = x

    return dxdt, dydt


# ---------------------------------------------------------------------------
# Numerical integration
# ---------------------------------------------------------------------------

def rk4_step(
    x: float,
    y: float,
    dt: float,
) -> tuple[float, float]:
    """Advance the microscopic system by one RK4 step."""

    k1x, k1y = micro_rhs(x, y)

    k2x, k2y = micro_rhs(
        x + 0.5 * dt * k1x,
        y + 0.5 * dt * k1y,
    )

    k3x, k3y = micro_rhs(
        x + 0.5 * dt * k2x,
        y + 0.5 * dt * k2y,
    )

    k4x, k4y = micro_rhs(
        x + dt * k3x,
        y + dt * k3y,
    )

    x_new = x + (dt / 6.0) * (
        k1x
        + 2.0 * k2x
        + 2.0 * k3x
        + k4x
    )

    y_new = y + (dt / 6.0) * (
        k1y
        + 2.0 * k2y
        + 2.0 * k3y
        + k4y
    )

    return x_new, y_new


def integrate_micro(
    x0: float,
    y0: float,
    dt: float,
    t_final: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the full microscopic system."""

    n_steps = int(round(t_final / dt))

    times = np.arange(
        n_steps + 1,
        dtype=float,
    ) * dt

    x = np.empty(n_steps + 1)
    y = np.empty(n_steps + 1)

    x[0] = x0
    y[0] = y0

    for i in range(n_steps):
        x[i + 1], y[i + 1] = rk4_step(
            x[i],
            y[i],
            dt,
        )

    return times, x, y


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_x(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Project the microscopic state onto the observed variable x.

    The hidden variable y is discarded.
    """

    del y

    return np.asarray(x)


# ---------------------------------------------------------------------------
# Hidden-state closure diagnostic
# ---------------------------------------------------------------------------

def estimate_dxdt(
    x: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Estimate dx/dt from a sampled trajectory."""

    return np.gradient(
        x,
        times,
    )


def bin_closure_relation(
    x: np.ndarray,
    dxdt: np.ndarray,
    n_bins: int,
) -> dict[str, np.ndarray]:
    """
    Estimate the relationship between observed x and dx/dt.

    If x alone is dynamically closed, points with similar x should have
    similar dx/dt.

    For this microscopic system:

        dx/dt = -y

    so the same x can correspond to different derivatives depending
    on the hidden state y.
    """

    xmin = float(np.min(x))
    xmax = float(np.max(x))

    edges = np.linspace(
        xmin,
        xmax,
        n_bins + 1,
    )

    centers = 0.5 * (
        edges[:-1]
        + edges[1:]
    )

    mean_dxdt = np.full(n_bins, np.nan)
    std_dxdt = np.full(n_bins, np.nan)
    count = np.zeros(n_bins, dtype=int)

    indices = np.digitize(
        x,
        edges,
    ) - 1

    for i in range(n_bins):
        mask = indices == i

        if np.any(mask):
            values = dxdt[mask]

            mean_dxdt[i] = np.mean(values)
            std_dxdt[i] = np.std(values)
            count[i] = np.sum(mask)

    return {
        "centers": centers,
        "mean_dxdt": mean_dxdt,
        "std_dxdt": std_dxdt,
        "count": count,
    }


# ---------------------------------------------------------------------------
# Naive effective closure
# ---------------------------------------------------------------------------

def naive_effective_rhs(
    x: float,
) -> float:
    """
    A deliberately incomplete x-only closure.

    We take the mean hidden-state contribution as zero:

        dx/dt = -E[y | x] ≈ 0.

    Therefore the naive closure predicts:

        dx/dt = 0.

    This is intentionally simplistic. It is NOT claimed to be the correct
    effective theory.

    Its purpose is to show that discarding y can destroy predictive power.
    """

    return 0.0


def integrate_naive_effective(
    x0: float,
    dt: float,
    t_final: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the naive x-only closure."""

    n_steps = int(round(t_final / dt))

    times = np.arange(
        n_steps + 1,
        dtype=float,
    ) * dt

    x = np.empty(n_steps + 1)

    x[0] = x0

    for i in range(n_steps):
        x[i + 1] = (
            x[i]
            + dt * naive_effective_rhs(x[i])
        )

    return times, x


# ---------------------------------------------------------------------------
# Singular closure experiment
# ---------------------------------------------------------------------------

def singular_effective_rhs(
    x: float,
    xc: float = 0.75,
) -> float:
    """
    A reduced closure generated from a reciprocal approximation.

    This is NOT inserted into the microscopic system.

    The purpose is to test whether a singular reduced law can coexist
    mathematically with a completely regular underlying system.

    The closure is:

        dx/dt = 1 / (xc - x)

    for x < xc.

    As x approaches xc, the derivative diverges.

    This provides a controlled comparison with the regular microscopic
    dynamics.

    IMPORTANT:
    This function is a diagnostic construction, not a physical claim.
    """

    denominator = xc - x

    if denominator <= 0.0:
        return np.inf

    return 1.0 / denominator


def integrate_singular_closure(
    x0: float,
    dt: float,
    t_final: float,
    xc: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, float | None]:
    """
    Integrate the singular effective closure.

    Returns
    -------
    times
    x
    blowup_time

    The blow-up time is estimated from the first numerical crossing
    of xc.
    """

    n_steps = int(round(t_final / dt))

    times = np.arange(
        n_steps + 1,
        dtype=float,
    ) * dt

    x = np.empty(n_steps + 1)

    x[0] = x0

    blowup_time = None

    for i in range(n_steps):

        rhs = singular_effective_rhs(
            x[i],
            xc=xc,
        )

        if not np.isfinite(rhs):
            blowup_time = times[i]
            x[i + 1 :] = np.nan
            break

        x_next = x[i] + dt * rhs

        x[i + 1] = x_next

        if x_next >= xc:
            blowup_time = times[i + 1]
            x[i + 1 :] = np.nan
            break

    return times, x, blowup_time


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class RunSummary:
    """Summary statistics for one microscopic trajectory."""

    y0: float
    max_radius_error: float
    max_abs_x: float
    max_abs_y: float
    closure_std_mean: float
    closure_std_max: float


def analyse_run(
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int,
) -> RunSummary:

    dxdt = estimate_dxdt(
        x,
        times,
    )

    relation = bin_closure_relation(
        x,
        dxdt,
        n_bins,
    )

    valid = np.isfinite(
        relation["std_dxdt"]
    )

    if np.any(valid):
        closure_std_mean = float(
            np.mean(
                relation["std_dxdt"][valid]
            )
        )

        closure_std_max = float(
            np.max(
                relation["std_dxdt"][valid]
            )
        )

    else:
        closure_std_mean = 0.0
        closure_std_max = 0.0

    radius = np.sqrt(
        x**2 + y**2
    )

    radius_error = np.max(
        np.abs(radius - radius[0])
    )

    return RunSummary(
        y0=float(y[0]),
        max_radius_error=float(radius_error),
        max_abs_x=float(np.max(np.abs(x))),
        max_abs_y=float(np.max(np.abs(y))),
        closure_std_mean=closure_std_mean,
        closure_std_max=closure_std_max,
    )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_runs = []

    print("=" * 72)
    print("Experiment 006: Regular Microdynamics -> Singular Effective Model")
    print("=" * 72)
    print()

    # ---------------------------------------------------------------
    # Full microscopic runs
    # ---------------------------------------------------------------

    for y0 in Y_VARIANTS:

        radius0 = np.sqrt(
            X0**2 + y0**2
        )

        # Only use initial states inside the unit disk.
        # For consistency, rescale each state onto the unit circle.
        scale = 1.0 / max(radius0, 1.0)

        x_initial = X0 * scale
        y_initial = y0 * scale

        times, x, y = integrate_micro(
            x_initial,
            y_initial,
            DT,
            T_FINAL,
        )

        summary = analyse_run(
            times,
            x,
            y,
            N_BINS,
        )

        all_runs.append(
            (
                times,
                x,
                y,
                summary,
            )
        )

        print(
            f"y0={y_initial:+.6f}  "
            f"max|x|={summary.max_abs_x:.6f}  "
            f"max|y|={summary.max_abs_y:.6f}  "
            f"radius_error={summary.max_radius_error:.3e}  "
            f"closure_std_max={summary.closure_std_max:.6f}"
        )

    print()

    # ---------------------------------------------------------------
    # Naive effective model
    # ---------------------------------------------------------------

    times_eff, x_eff = integrate_naive_effective(
        X0,
        DT,
        T_FINAL,
    )

    # ---------------------------------------------------------------
    # Singular effective model
    # ---------------------------------------------------------------

    times_sing, x_sing, blowup_time = (
        integrate_singular_closure(
            X0,
            DT,
            T_FINAL,
            xc=0.75,
        )
    )

    if blowup_time is None:
        blowup_text = "not observed"
    else:
        blowup_text = f"{blowup_time:.6f}"

    print(
        "Naive effective model:"
    )
    print(
        f"  initial x = {X0:.6f}"
    )
    print(
        f"  final x   = {x_eff[-1]:.6f}"
    )
    print()

    print(
        "Singular effective model:"
    )
    print(
        f"  threshold xc = 0.750000"
    )
    print(
        f"  blow-up / threshold crossing = {blowup_text}"
    )
    print()

    # ---------------------------------------------------------------
    # CSV output
    # ---------------------------------------------------------------

    csv_path = (
        RESULTS_DIR
        / "006_projection_singularity.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.writer(handle)

        writer.writerow(
            [
                "y0",
                "max_abs_x",
                "max_abs_y",
                "max_radius_error",
                "closure_std_mean",
                "closure_std_max",
            ]
        )

        for _, _, _, summary in all_runs:

            writer.writerow(
                [
                    summary.y0,
                    summary.max_abs_x,
                    summary.max_abs_y,
                    summary.max_radius_error,
                    summary.closure_std_mean,
                    summary.closure_std_max,
                ]
            )

    # ---------------------------------------------------------------
    # Summary output
    # ---------------------------------------------------------------

    summary_path = (
        RESULTS_DIR
        / "006_projection_singularity_summary.txt"
    )

    with summary_path.open(
        "w"
    ) as handle:

        handle.write(
            "Experiment 006: Regular Microdynamics -> "
            "Singular Effective Description\n"
        )

        handle.write("=" * 72 + "\n\n")

        handle.write(
            "Question:\n"
        )

        handle.write(
            "Can bounded underlying dynamics produce an apparently "
            "singular effective description after projection?\n\n"
        )

        handle.write(
            "Microscopic system:\n"
        )

        handle.write(
            "    dx/dt = -y\n"
            "    dy/dt =  x\n\n"
        )

        handle.write(
            "Projection:\n"
        )

        handle.write(
            "    C(x,y) = x\n\n"
        )

        handle.write(
            "The microscopic system remains bounded because\n"
            "x^2 + y^2 is conserved.\n\n"
        )

        handle.write(
            "Naive effective closure:\n"
            "    dx/dt = 0\n\n"
        )

        handle.write(
            "Diagnostic singular closure:\n"
            "    dx/dt = 1 / (xc - x)\n"
            "    xc = 0.75\n\n"
        )

        handle.write(
            f"Singular closure threshold crossing: "
            f"{blowup_text}\n\n"
        )

        handle.write(
            "Interpretation:\n"
        )

        handle.write(
            "The experiment distinguishes two ideas.\n\n"
        )

        handle.write(
            "1. Projection can destroy dynamic closure. "
            "The same observed x can correspond to different dx/dt "
            "because y has been discarded.\n\n"
        )

        handle.write(
            "2. A singular effective law is not automatically evidence "
            "of a singular underlying system. A reduced description "
            "can contain pathology that is absent from the complete "
            "microscopic dynamics.\n\n"
        )

        handle.write(
            "IMPORTANT:\n"
        )

        handle.write(
            "The singular closure in this experiment is deliberately "
            "constructed as a diagnostic. It is not derived from the "
            "microscopic equations. Therefore this experiment does "
            "NOT yet demonstrate that projection naturally generates "
            "a singularity.\n\n"
        )

        handle.write(
            "The next step, if warranted, is to construct an effective "
            "closure directly from the projected data and investigate "
            "whether the loss of hidden-state information can itself "
            "generate a singular term.\n"
        )

    # ---------------------------------------------------------------
    # Plot 1: microscopic trajectories
    # ---------------------------------------------------------------

    fig = plt.figure(
        figsize=(10, 6)
    )

    for times, x, y, summary in all_runs:

        plt.plot(
            times,
            x,
            label=f"y0={summary.y0:+.2f}",
        )

    plt.xlabel("time")
    plt.ylabel("observed x(t)")
    plt.title(
        "Projected microscopic trajectories"
    )
    plt.legend()
    plt.tight_layout()

    fig.savefig(
        RESULTS_DIR
        / "006_micro_and_projection.png",
        dpi=160,
    )

    plt.close(fig)

    # ---------------------------------------------------------------
    # Plot 2: effective closures
    # ---------------------------------------------------------------

    fig = plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        times_eff,
        x_eff,
        label="naive x-only closure",
    )

    plt.plot(
        times_sing,
        x_sing,
        label="diagnostic singular closure",
    )

    plt.axhline(
        0.75,
        linestyle="--",
        label="singular threshold",
    )

    plt.xlabel("time")
    plt.ylabel("effective x(t)")
    plt.title(
        "Effective descriptions"
    )
    plt.legend()
    plt.tight_layout()

    fig.savefig(
        RESULTS_DIR
        / "006_effective_closure.png",
        dpi=160,
    )

    plt.close(fig)

    # ---------------------------------------------------------------
    # Plot 3: microscopic phase space
    # ---------------------------------------------------------------

    fig = plt.figure(
        figsize=(7, 7)
    )

    for times, x, y, summary in all_runs:

        plt.plot(
            x,
            y,
            label=f"y0={summary.y0:+.2f}",
        )

    plt.xlabel("x")
    plt.ylabel("hidden y")
    plt.title(
        "Bounded microscopic phase space"
    )
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()

    fig.savefig(
        RESULTS_DIR
        / "006_phase_space.png",
        dpi=160,
    )

    plt.close(fig)

    print(
        "Results written to:"
    )

    print(
        f"  {csv_path}"
    )

    print(
        f"  {summary_path}"
    )

    print(
        "  results/006_micro_and_projection.png"
    )

    print(
        "  results/006_effective_closure.png"
    )

    print(
        "  results/006_phase_space.png"
    )


if __name__ == "__main__":
    main()
