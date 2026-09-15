"""
Experiment 005g: Local vs Nonlocal Closure

Purpose
-------
005f introduced an explicit microscopic spatial scale a through a nonlocal
BGK relaxation operator:

    df/dt + v df/dx = [f_eq[bar_f_a] - f] / tau

005g asks whether that nonlocal term actually changes the dynamics in a
measurable way.

We compare:

    LOCAL:
        f_eq[f]

    NONLOCAL:
        f_eq[bar_f_a]

The central diagnostic is the difference between the two dynamical
operators:

    D_a[f] = F_a[f] - F_0[f]

where:

    F_0 = local BGK operator
    F_a = nonlocal BGK operator

We also retain the hidden-state A/B experiment from 005f and compare the
resulting hydrodynamic discrepancy.

This experiment therefore separates two questions:

    1. Does the microscopic length a modify the dynamics?

    2. Does that modification become more important as L/a decreases?

This is still a toy kinetic model. It is NOT a derivation of Navier-Stokes.

Outputs
-------
results/005g_local_vs_nonlocal.csv
results/005g_local_vs_nonlocal_summary.txt

results/005g_operator_difference_vs_L_over_a.png
results/005g_operator_difference_vs_a.png
results/005g_closure_comparison.png
results/005g_operator_heatmap.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NX = 256
NV = 96

X_MIN = 0.0
X_MAX = 2.0 * np.pi

V_MIN = -4.0
V_MAX = 4.0

DX = (X_MAX - X_MIN) / NX
DV = (V_MAX - V_MIN) / NV

DT = 0.001
T_FINAL = 4.0

# Macroscopic wavelengths.
L_VALUES = np.array(
    [
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
    ],
    dtype=float,
)

# Explicit microscopic scales.
A_VALUES = np.array(
    [
        0.025,
        0.05,
        0.10,
        0.20,
    ],
    dtype=float,
)

# Relaxation times.
TAU_VALUES = np.array(
    [
        0.01,
        0.05,
        0.10,
    ],
    dtype=float,
)

HIDDEN_AMPLITUDE = 0.35

DENSITY_FLOOR = 1.0e-12
TEMPERATURE_FLOOR = 1.0e-12


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


def make_grid() -> tuple[np.ndarray, np.ndarray]:
    """Create spatial and velocity grids."""

    x = np.arange(
        NX,
        dtype=float,
    ) * DX

    v = (
        V_MIN
        + (np.arange(NV, dtype=float) + 0.5) * DV
    )

    return x, v


# ---------------------------------------------------------------------------
# Maxwellian
# ---------------------------------------------------------------------------


def maxwellian(
    rho: np.ndarray,
    u: np.ndarray,
    temperature: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """Construct a 1-D velocity Maxwellian."""

    rho = np.maximum(
        rho,
        DENSITY_FLOOR,
    )

    temperature = np.maximum(
        temperature,
        TEMPERATURE_FLOOR,
    )

    normalization = np.sqrt(
        2.0 * np.pi * temperature
    )

    exponent = -(
        v[None, :] - u[:, None]
    ) ** 2 / (
        2.0 * temperature[:, None]
    )

    return (
        rho[:, None]
        / normalization[:, None]
        * np.exp(exponent)
    )


# ---------------------------------------------------------------------------
# Hydrodynamic moments
# ---------------------------------------------------------------------------


def hydro_moments(
    f: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate density, velocity and temperature.

    Returns
    -------
    rho
    u
    temperature
    """

    rho = (
        np.sum(
            f,
            axis=1,
        )
        * DV
    )

    rho_safe = np.maximum(
        rho,
        DENSITY_FLOOR,
    )

    momentum = (
        np.sum(
            f * v[None, :],
            axis=1,
        )
        * DV
    )

    u = momentum / rho_safe

    thermal_energy = (
        np.sum(
            f
            * (
                v[None, :]
                - u[:, None]
            ) ** 2,
            axis=1,
        )
        * DV
    )

    temperature = (
        thermal_energy
        / rho_safe
    )

    temperature = np.maximum(
        temperature,
        TEMPERATURE_FLOOR,
    )

    return (
        rho,
        u,
        temperature,
    )


# ---------------------------------------------------------------------------
# Initial macroscopic state
# ---------------------------------------------------------------------------


def initial_hydro_state(
    x: np.ndarray,
    L: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create the initial macroscopic perturbation."""

    mode = 2.0 * np.pi / L

    rho = (
        1.0
        + 0.05 * np.cos(mode * x)
    )

    u = (
        0.35 * np.sin(mode * x)
    )

    temperature = (
        1.0
        + 0.05 * np.cos(mode * x)
    )

    return (
        rho,
        u,
        temperature,
    )


# ---------------------------------------------------------------------------
# Hidden kinetic state
# ---------------------------------------------------------------------------


def hidden_kinetic_shape(
    v: np.ndarray,
) -> np.ndarray:
    """
    Construct a higher-order velocity-space perturbation.

    The perturbation is approximately orthogonal to the hydrodynamic
    moments 1, v and v^2.
    """

    z = v / np.sqrt(2.0)

    h4 = (
        z**4
        - 6.0 * z**2
        + 3.0
    )

    weight = np.exp(
        -0.5 * v**2
    )

    shape = h4 * weight

    basis = np.vstack(
        [
            np.ones_like(v),
            v,
            v**2,
        ]
    ).T

    weighted_basis = (
        basis
        * weight[:, None]
    )

    coefficients, *_ = np.linalg.lstsq(
        weighted_basis,
        shape,
        rcond=None,
    )

    correction = (
        basis @ coefficients
    )

    shape = (
        shape
        - correction * weight
    )

    maximum = np.max(
        np.abs(shape)
    )

    if maximum > 0.0:
        shape /= maximum

    return shape


# ---------------------------------------------------------------------------
# Initial distributions
# ---------------------------------------------------------------------------


def make_initial_distributions(
    x: np.ndarray,
    v: np.ndarray,
    L: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create two states A and B with nearly identical hydrodynamic moments.

    A = local Maxwellian

    B = Maxwellian plus higher-order kinetic structure.
    """

    rho, u, temperature = (
        initial_hydro_state(
            x,
            L,
        )
    )

    equilibrium = maxwellian(
        rho,
        u,
        temperature,
        v,
    )

    velocity_shape = (
        hidden_kinetic_shape(v)
    )

    spatial_shape = np.cos(
        (2.0 * np.pi / L) * x
    )

    perturbation = (
        HIDDEN_AMPLITUDE
        * equilibrium
        * spatial_shape[:, None]
        * velocity_shape[None, :]
    )

    f_a = equilibrium.copy()

    f_b = equilibrium + perturbation

    f_b = np.maximum(
        f_b,
        1.0e-14,
    )

    return (
        f_a,
        f_b,
    )


# ---------------------------------------------------------------------------
# Spatial filtering
# ---------------------------------------------------------------------------


def gaussian_kernel(
    a: float,
) -> np.ndarray:
    """Create a normalized periodic Gaussian kernel of width a."""

    if a <= 0.0:
        raise ValueError(
            "a must be positive."
        )

    radius = max(
        1,
        int(
            np.ceil(
                4.0 * a / DX
            )
        ),
    )

    offsets = np.arange(
        -radius,
        radius + 1,
        dtype=float,
    )

    distances = offsets * DX

    kernel = np.exp(
        -0.5
        * (
            distances / a
        ) ** 2
    )

    kernel /= np.sum(
        kernel
    )

    return kernel


def periodic_filter(
    f: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """Apply a periodic spatial convolution."""

    filtered = np.zeros_like(
        f
    )

    centre = len(kernel) // 2

    for index, weight in enumerate(
        kernel
    ):
        shift = (
            index
            - centre
        )

        filtered += (
            weight
            * np.roll(
                f,
                shift=shift,
                axis=0,
            )
        )

    return filtered


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def streaming_rhs(
    f: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """
    First-order upwind streaming operator.

        -v df/dx
    """

    streaming = np.zeros_like(
        f
    )

    for j, velocity in enumerate(
        v
    ):
        if velocity >= 0.0:
            derivative = (
                f[:, j]
                - np.roll(
                    f[:, j],
                    1,
                )
            ) / DX
        else:
            derivative = (
                np.roll(
                    f[:, j],
                    -1,
                )
                - f[:, j]
            ) / DX

        streaming[:, j] = (
            -velocity
            * derivative
        )

    return streaming


# ---------------------------------------------------------------------------
# Local BGK operator
# ---------------------------------------------------------------------------


def local_bgk_rhs(
    f: np.ndarray,
    v: np.ndarray,
    tau: float,
) -> np.ndarray:
    """
    Local BGK dynamics.

        df/dt + v df/dx
        =
        [f_eq[f] - f] / tau
    """

    streaming = streaming_rhs(
        f,
        v,
    )

    rho, u, temperature = (
        hydro_moments(
            f,
            v,
        )
    )

    equilibrium = maxwellian(
        rho,
        u,
        temperature,
        v,
    )

    relaxation = (
        equilibrium - f
    ) / tau

    return (
        streaming
        + relaxation
    )


# ---------------------------------------------------------------------------
# Nonlocal BGK operator
# ---------------------------------------------------------------------------


def nonlocal_bgk_rhs(
    f: np.ndarray,
    v: np.ndarray,
    tau: float,
    kernel: np.ndarray,
) -> np.ndarray:
    """
    Nonlocal BGK dynamics.

        df/dt + v df/dx
        =
        [f_eq[bar(f)] - f] / tau
    """

    streaming = streaming_rhs(
        f,
        v,
    )

    filtered = periodic_filter(
        f,
        kernel,
    )

    rho, u, temperature = (
        hydro_moments(
            filtered,
            v,
        )
    )

    equilibrium = maxwellian(
        rho,
        u,
        temperature,
        v,
    )

    relaxation = (
        equilibrium - f
    ) / tau

    return (
        streaming
        + relaxation
    )


# ---------------------------------------------------------------------------
# Operator diagnostics
# ---------------------------------------------------------------------------


def operator_difference(
    f: np.ndarray,
    v: np.ndarray,
    tau: float,
    kernel: np.ndarray,
) -> tuple[float, float, float]:
    """
    Compare local and nonlocal dynamical operators.

    Returns:

        absolute L2 difference
        relative L2 difference
        RMS difference
    """

    local = local_bgk_rhs(
        f,
        v,
        tau,
    )

    nonlocal = nonlocal_bgk_rhs(
        f,
        v,
        tau,
        kernel,
    )

    difference = (
        nonlocal
        - local
    )

    absolute = np.sqrt(
        np.mean(
            difference**2
        )
    )

    local_norm = np.sqrt(
        np.mean(
            local**2
        )
    )

    relative = (
        absolute
        / max(
            local_norm,
            1.0e-14,
        )
    )

    rms = np.sqrt(
        np.mean(
            difference**2
        )
    )

    return (
        absolute,
        relative,
        rms,
    )


# ---------------------------------------------------------------------------
# Hidden-state hydrodynamic discrepancy
# ---------------------------------------------------------------------------


def hydrodynamic_discrepancy(
    f_a: np.ndarray,
    f_b: np.ndarray,
    v: np.ndarray,
) -> tuple[float, float, float, float]:
    """Calculate hydrodynamic discrepancy between A and B."""

    rho_a, u_a, temp_a = (
        hydro_moments(
            f_a,
            v,
        )
    )

    rho_b, u_b, temp_b = (
        hydro_moments(
            f_b,
            v,
        )
    )

    density = np.sqrt(
        np.mean(
            (rho_a - rho_b) ** 2
        )
    )

    velocity = np.sqrt(
        np.mean(
            (u_a - u_b) ** 2
        )
    )

    temperature = np.sqrt(
        np.mean(
            (temp_a - temp_b) ** 2
        )
    )

    combined = np.sqrt(
        density**2
        + velocity**2
        + temperature**2
    )

    return (
        density,
        velocity,
        temperature,
        combined,
    )


# ---------------------------------------------------------------------------
# RK2
# ---------------------------------------------------------------------------


def rk2_local(
    f: np.ndarray,
    v: np.ndarray,
    dt: float,
    tau: float,
) -> np.ndarray:
    """RK2 step for local BGK."""

    k1 = local_bgk_rhs(
        f,
        v,
        tau,
    )

    midpoint = (
        f
        + 0.5 * dt * k1
    )

    k2 = local_bgk_rhs(
        midpoint,
        v,
        tau,
    )

    return np.maximum(
        f + dt * k2,
        1.0e-14,
    )


def rk2_nonlocal(
    f: np.ndarray,
    v: np.ndarray,
    dt: float,
    tau: float,
    kernel: np.ndarray,
) -> np.ndarray:
    """RK2 step for nonlocal BGK."""

    k1 = nonlocal_bgk_rhs(
        f,
        v,
        tau,
        kernel,
    )

    midpoint = (
        f
        + 0.5 * dt * k1
    )

    k2 = nonlocal_bgk_rhs(
        midpoint,
        v,
        tau,
        kernel,
    )

    return np.maximum(
        f + dt * k2,
        1.0e-14,
    )


# ---------------------------------------------------------------------------
# Single case
# ---------------------------------------------------------------------------


def run_case(
    x: np.ndarray,
    v: np.ndarray,
    L: float,
    a: float,
    tau: float,
) -> dict[str, float]:
    """
    Run one local/nonlocal comparison.

    We evolve the SAME initial state under both operators. This isolates the
    effect of the nonlocal term itself.
    """

    f_local, _ = make_initial_distributions(
        x,
        v,
        L,
    )

    f_nonlocal = f_local.copy()

    kernel = gaussian_kernel(
        a
    )

    # Initial operator difference.
    op_initial = operator_difference(
        f_local,
        v,
        tau,
        kernel,
    )

    initial_operator = op_initial[0]
    initial_operator_relative = op_initial[1]

    initial_discrepancy = (
        hydrodynamic_discrepancy(
            f_local,
            f_nonlocal,
            v,
        )[3]
    )

    n_steps = int(
        round(
            T_FINAL / DT
        )
    )

    maximum_state_difference = 0.0
    maximum_state_time = 0.0

    maximum_operator_difference = (
        initial_operator
    )

    maximum_operator_time = 0.0

    final_operator = initial_operator
    final_operator_relative = (
        initial_operator_relative
    )

    for step in range(
        1,
        n_steps + 1,
    ):

        # Operator evaluated on the local state before advancing.
        op = operator_difference(
            f_local,
            v,
            tau,
            kernel,
        )

        if op[0] > maximum_operator_difference:
            maximum_operator_difference = (
                op[0]
            )

            maximum_operator_time = (
                step * DT
            )

        f_local = rk2_local(
            f_local,
            v,
            DT,
            tau,
        )

        f_nonlocal = rk2_nonlocal(
            f_nonlocal,
            v,
            DT,
            tau,
            kernel,
        )

        state_difference = (
            np.sqrt(
                np.mean(
                    (
                        f_local
                        - f_nonlocal
                    ) ** 2
                )
            )
        )

        if (
            state_difference
            > maximum_state_difference
        ):
            maximum_state_difference = (
                state_difference
            )

            maximum_state_time = (
                step * DT
            )

        final_operator = op[0]
        final_operator_relative = op[1]

    final_hydro = (
        hydrodynamic_discrepancy(
            f_local,
            f_nonlocal,
            v,
        )
    )

    L_over_a = L / a

    return {
        "L": L,
        "a": a,
        "L_over_a": L_over_a,
        "tau": tau,
        "initial_operator_difference": initial_operator,
        "initial_operator_relative": initial_operator_relative,
        "final_operator_difference": final_operator,
        "final_operator_relative": final_operator_relative,
        "maximum_operator_difference": (
            maximum_operator_difference
        ),
        "time_of_max_operator_difference": (
            maximum_operator_time
        ),
        "initial_local_nonlocal_state_difference": (
            initial_discrepancy
        ),
        "final_density_difference": final_hydro[0],
        "final_velocity_difference": final_hydro[1],
        "final_temperature_difference": final_hydro[2],
        "final_combined_hydro_difference": (
            final_hydro[3]
        ),
        "maximum_local_nonlocal_state_difference": (
            maximum_state_difference
        ),
        "time_of_max_state_difference": (
            maximum_state_time
        ),
    }


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def write_csv(
    rows: list[dict[str, float]],
    path: Path,
) -> None:
    """Write results to CSV."""

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def write_summary(
    rows: list[dict[str, float]],
    path: Path,
) -> None:
    """Write human-readable analysis."""

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "Experiment 005g: Local vs Nonlocal Closure\n"
        )

        handle.write(
            "=" * 68
            + "\n\n"
        )

        handle.write(
            "The experiment compares local BGK dynamics with a nonlocal "
            "BGK model containing an explicit microscopic length a.\n\n"
        )

        handle.write(
            "Primary diagnostic:\n"
            "    D_a[f] = F_a[f] - F_0[f]\n\n"
        )

        handle.write(
            "where F_0 is the local kinetic operator and F_a is the "
            "nonlocal operator.\n\n"
        )

        handle.write(
            f"NX={NX}, NV={NV}\n"
        )

        handle.write(
            f"DX={DX:.8f}, DV={DV:.8f}\n"
        )

        handle.write(
            f"DT={DT}, T_FINAL={T_FINAL}\n\n"
        )

        # Global ranges.
        operator_values = np.array(
            [
                row[
                    "maximum_operator_difference"
                ]
                for row in rows
            ]
        )

        hydro_values = np.array(
            [
                row[
                    "final_combined_hydro_difference"
                ]
                for row in rows
            ]
        )

        handle.write(
            "Global ranges\n"
        )

        handle.write(
            "-" * 68
            + "\n"
        )

        handle.write(
            "Maximum operator difference: "
            f"{operator_values.min():.8e} "
            f"to {operator_values.max():.8e}\n"
        )

        handle.write(
            "Final hydrodynamic difference: "
            f"{hydro_values.min():.8e} "
            f"to {hydro_values.max():.8e}\n\n"
        )

        # Group by tau.
        handle.write(
            "Operator difference grouped by tau\n"
        )

        handle.write(
            "-" * 68
            + "\n"
        )

        for tau in TAU_VALUES:

            subset = [
                row
                for row in rows
                if np.isclose(
                    row["tau"],
                    tau,
                )
            ]

            subset.sort(
                key=lambda row:
                row["L_over_a"]
            )

            handle.write(
                f"\ntau={tau:g}\n"
            )

            handle.write(
                "  L/a       L       a       "
                "max_operator      final_hydro\n"
            )

            for row in subset:

                handle.write(
                    f"  "
                    f"{row['L_over_a']:8.3f} "
                    f"{row['L']:7.3f} "
                    f"{row['a']:7.3f} "
                    f"{row['maximum_operator_difference']:14.7e} "
                    f"{row['final_combined_hydro_difference']:14.7e}\n"
                )

        # Equal L/a.
        handle.write(
            "\n\nEqual L/a comparisons\n"
        )

        handle.write(
            "-" * 68
            + "\n"
        )

        ratios = sorted(
            {
                round(
                    row["L_over_a"],
                    10,
                )
                for row in rows
            }
        )

        for ratio in ratios:

            subset = [
                row
                for row in rows
                if np.isclose(
                    row["L_over_a"],
                    ratio,
                )
            ]

            if len(subset) < 2:
                continue

            values = np.array(
                [
                    row[
                        "maximum_operator_difference"
                    ]
                    for row in subset
                ]
            )

            handle.write(
                f"L/a={ratio:.5f}: "
                f"operator min={values.min():.8e}, "
                f"max={values.max():.8e}, "
                f"spread={values.max() - values.min():.8e}\n"
            )

        handle.write(
            "\n\nInterpretation\n"
        )

        handle.write(
            "-" * 68
            + "\n"
        )

        handle.write(
            "A clear increase in operator difference as L/a decreases "
            "would indicate that the explicit microscopic length is "
            "becoming dynamically important near the scale boundary.\n\n"
        )

        handle.write(
            "If the operator difference remains small while the "
            "hydrodynamic trajectories diverge, the effect may arise "
            "through long-time accumulation rather than an instantaneous "
            "change of the kinetic operator.\n\n"
        )

        handle.write(
            "If both remain weakly dependent on L/a, the particular "
            "nonlocal mechanism used here does not produce a strong "
            "scale-boundary effect.\n\n"
        )

        handle.write(
            "None of these outcomes establishes anything about the "
            "Navier-Stokes singularity. The experiment tests a generic "
            "multiscale closure mechanism.\n"
        )


# ---------------------------------------------------------------------------
# Plot 1: operator difference vs L/a
# ---------------------------------------------------------------------------


def plot_operator_vs_ratio(
    rows: list[dict[str, float]],
) -> None:
    """Plot operator difference against L/a."""

    plt.figure()

    for tau in TAU_VALUES:

        subset = [
            row
            for row in rows
            if np.isclose(
                row["tau"],
                tau,
            )
        ]

        subset.sort(
            key=lambda row:
            row["L_over_a"]
        )

        x = np.array(
            [
                row["L_over_a"]
                for row in subset
            ]
        )

        y = np.array(
            [
                row[
                    "maximum_operator_difference"
                ]
                for row in subset
            ]
        )

        plt.plot(
            x,
            y,
            marker="o",
            label=f"tau={tau:g}",
        )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(
        "Macroscopic / microscopic scale L/a"
    )

    plt.ylabel(
        "Maximum operator difference"
    )

    plt.title(
        "005g: Local vs nonlocal dynamics"
    )

    plt.legend()
    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "005g_operator_difference_vs_L_over_a.png",
        dpi=160,
    )

    plt.close()


# ---------------------------------------------------------------------------
# Plot 2: operator difference vs a
# ---------------------------------------------------------------------------


def plot_operator_vs_a(
    rows: list[dict[str, float]],
) -> None:
    """Plot operator difference against microscopic scale."""

    plt.figure()

    selected_tau = TAU_VALUES[1]

    for L in L_VALUES:

        subset = [
            row
            for row in rows
            if np.isclose(
                row["L"],
                L,
            )
            and np.isclose(
                row["tau"],
                selected_tau,
            )
        ]

        subset.sort(
            key=lambda row:
            row["a"]
        )

        x = np.array(
            [
                row["a"]
                for row in subset
            ]
        )

        y = np.array(
            [
                row[
                    "maximum_operator_difference"
                ]
                for row in subset
            ]
        )

        plt.plot(
            x,
            y,
            marker="o",
            label=f"L={L:g}",
        )

    plt.xlabel(
        "Microscopic scale a"
    )

    plt.ylabel(
        "Maximum operator difference"
    )

    plt.title(
        f"005g: Explicit microscopic-scale effect "
        f"(tau={selected_tau:g})"
    )

    plt.legend()
    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "005g_operator_difference_vs_a.png",
        dpi=160,
    )

    plt.close()


# ---------------------------------------------------------------------------
# Plot 3: hydrodynamic comparison
# ---------------------------------------------------------------------------


def plot_closure_comparison(
    rows: list[dict[str, float]],
) -> None:
    """Plot final local/nonlocal hydrodynamic separation."""

    plt.figure()

    for tau in TAU_VALUES:

        subset = [
            row
            for row in rows
            if np.isclose(
                row["tau"],
                tau,
            )
        ]

        subset.sort(
            key=lambda row:
            row["L_over_a"]
        )

        x = np.array(
            [
                row["L_over_a"]
                for row in subset
            ]
        )

        y = np.array(
            [
                row[
                    "final_combined_hydro_difference"
                ]
                for row in subset
            ]
        )

        plt.plot(
            x,
            y,
            marker="o",
            label=f"tau={tau:g}",
        )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(
        "Macroscopic / microscopic scale L/a"
    )

    plt.ylabel(
        "Final local/nonlocal hydrodynamic difference"
    )

    plt.title(
        "005g: Accumulated closure difference"
    )

    plt.legend()
    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "005g_closure_comparison.png",
        dpi=160,
    )

    plt.close()


# ---------------------------------------------------------------------------
# Plot 4: parameter map
# ---------------------------------------------------------------------------


def plot_operator_heatmap(
    rows: list[dict[str, float]],
) -> None:
    """Create a parameter map using L/a and tau."""

    plt.figure()

    x = np.array(
        [
            row["L_over_a"]
            for row in rows
        ]
    )

    y = np.array(
        [
            row["tau"]
            for row in rows
        ]
    )

    z = np.array(
        [
            row[
                "maximum_operator_difference"
            ]
            for row in rows
        ]
    )

    scatter = plt.scatter(
        x,
        y,
        c=z,
        s=70,
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel(
        "Macroscopic / microscopic scale L/a"
    )

    plt.ylabel(
        "Relaxation time tau"
    )

    plt.title(
        "005g: Dynamical effect of microscopic scale"
    )

    colorbar = plt.colorbar(
        scatter
    )

    colorbar.set_label(
        "Maximum operator difference"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "005g_operator_heatmap.png",
        dpi=160,
    )

    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run experiment 005g."""

    print("=" * 68)
    print(
        "Experiment 005g: Local vs Nonlocal Closure"
    )
    print("=" * 68)

    print()
    print(f"NX={NX}")
    print(f"NV={NV}")
    print(f"DX={DX:.8f}")
    print(f"DV={DV:.8f}")
    print(f"DT={DT}")
    print(f"T_FINAL={T_FINAL}")

    print()
    print("L values:", L_VALUES)
    print("a values:", A_VALUES)
    print("tau values:", TAU_VALUES)

    x, v = make_grid()

    rows: list[dict[str, float]] = []

    total_cases = (
        len(L_VALUES)
        * len(A_VALUES)
        * len(TAU_VALUES)
    )

    case_number = 0

    for L in L_VALUES:

        for a in A_VALUES:

            for tau in TAU_VALUES:

                case_number += 1

                print(
                    f"\n[{case_number}/{total_cases}] "
                    f"L={L:.6f}, "
                    f"a={a:.6f}, "
                    f"L/a={L/a:.3f}, "
                    f"tau={tau:.6f}"
                )

                result = run_case(
                    x,
                    v,
                    L,
                    a,
                    tau,
                )

                rows.append(
                    result
                )

                print(
                    "    initial operator="
                    f"{result['initial_operator_difference']:.8e}"
                )

                print(
                    "    maximum operator="
                    f"{result['maximum_operator_difference']:.8e}"
                )

                print(
                    "    final hydro difference="
                    f"{result['final_combined_hydro_difference']:.8e}"
                )

    csv_path = (
        RESULTS_DIR
        / "005g_local_vs_nonlocal.csv"
    )

    summary_path = (
        RESULTS_DIR
        / "005g_local_vs_nonlocal_summary.txt"
    )

    write_csv(
        rows,
        csv_path,
    )

    write_summary(
        rows,
        summary_path,
    )

    plot_operator_vs_ratio(
        rows
    )

    plot_operator_vs_a(
        rows
    )

    plot_closure_comparison(
        rows
    )

    plot_operator_heatmap(
        rows
    )

    print()
    print("=" * 68)
    print("Experiment complete.")
    print("=" * 68)

    print()
    print(f"CSV:     {csv_path}")
    print(f"Summary: {summary_path}")

    print()
    print("Plots:")
    print(
        "  results/005g_operator_difference_vs_L_over_a.png"
    )
    print(
        "  results/005g_operator_difference_vs_a.png"
    )
    print(
        "  results/005g_closure_comparison.png"
    )
    print(
        "  results/005g_operator_heatmap.png"
    )


if __name__ == "__main__":
    main()
