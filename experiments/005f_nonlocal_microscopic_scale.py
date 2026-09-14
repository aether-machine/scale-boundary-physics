"""
Experiment 005f: Explicit Microscopic Scale / Nonlocal BGK

Purpose
-------
Introduce an explicit microscopic spatial scale into the kinetic model and
test whether hydrodynamic closure error changes as the macroscopic scale L
approaches that microscopic scale a.

The model is a deliberately simple nonlocal BGK analogue:

    df/dt + v df/dx = [f_eq[bar_f_a] - f] / tau

where bar_f_a is a spatially smoothed version of f over a finite microscopic
length scale a.

This is NOT a derivation of Navier-Stokes.

The purpose is to test a narrower hypothesis:

    Can a coarse-grained macroscopic description become progressively less
    accurate as its characteristic scale approaches an explicit microscopic
    scale?

Primary dimensionless parameter:

    R = L / a

where:
    L = macroscopic spatial wavelength
    a = microscopic interaction/filter length

We also retain tau as an independent relaxation parameter.

The experiment compares two microscopic states A and B which have identical
initial hydrodynamic moments but differ in higher-order velocity structure.

The evolution of those hidden differences is then measured in the
hydrodynamic variables rho, u and T.

Outputs
-------
results/005f_nonlocal_microscopic_scale.csv
results/005f_nonlocal_microscopic_scale_summary.txt

results/005f_discrepancy_vs_L_over_a.png
results/005f_discrepancy_vs_microscopic_scale.png
results/005f_parameter_map.png
results/005f_discrepancy_vs_tau.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scale_boundary.metrics import state_discrepancy


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
#
# The largest wavelengths are comfortably above the microscopic scale.
# The smallest wavelengths approach the microscopic scale.
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

# Explicit microscopic spatial scales.
A_VALUES = np.array(
    [
        0.025,
        0.05,
        0.10,
        0.20,
    ],
    dtype=float,
)

# Relaxation times are deliberately varied independently of a and L.
TAU_VALUES = np.array(
    [
        0.01,
        0.05,
        0.10,
    ],
    dtype=float,
)

# Amplitude of the hidden kinetic perturbation.
HIDDEN_AMPLITUDE = 0.35

# Numerical / physical safety floors.
DENSITY_FLOOR = 1.0e-12
TEMPERATURE_FLOOR = 1.0e-12


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


def make_grid() -> tuple[np.ndarray, np.ndarray]:
    """Create periodic spatial grid and velocity grid."""

    x = np.arange(NX, dtype=float) * DX

    # Cell-centred velocity grid.
    v = V_MIN + (np.arange(NV, dtype=float) + 0.5) * DV

    return x, v


# ---------------------------------------------------------------------------
# Maxwellian / equilibrium
# ---------------------------------------------------------------------------


def maxwellian(
    rho: np.ndarray,
    u: np.ndarray,
    temperature: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """
    Construct a 1-D velocity Maxwellian.

    Returns
    -------
    f : ndarray, shape (NX, NV)
    """

    rho = np.maximum(rho, DENSITY_FLOOR)
    temperature = np.maximum(temperature, TEMPERATURE_FLOOR)

    thermal = np.sqrt(2.0 * np.pi * temperature)

    exponent = -(
        v[None, :] - u[:, None]
    ) ** 2 / (2.0 * temperature[:, None])

    return rho[:, None] / thermal[:, None] * np.exp(exponent)


# ---------------------------------------------------------------------------
# Hydrodynamic moments
# ---------------------------------------------------------------------------


def hydro_moments(
    f: np.ndarray,
    v: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recover density, velocity and temperature from f.

    The velocity integral is approximated using the uniform velocity grid.
    """

    rho = np.sum(f, axis=1) * DV
    rho_safe = np.maximum(rho, DENSITY_FLOOR)

    momentum = np.sum(f * v[None, :], axis=1) * DV

    u = momentum / rho_safe

    energy = (
        np.sum(
            f * (v[None, :] - u[:, None]) ** 2,
            axis=1,
        )
        * DV
    )

    temperature = energy / rho_safe

    temperature = np.maximum(
        temperature,
        TEMPERATURE_FLOOR,
    )

    return rho, u, temperature


# ---------------------------------------------------------------------------
# Initial macroscopic state
# ---------------------------------------------------------------------------


def initial_hydro_state(
    x: np.ndarray,
    L: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct a smooth macroscopic state with wavelength L.

    L is the physical wavelength of the imposed perturbation.
    """

    mode = 2.0 * np.pi / L

    rho = 1.0 + 0.05 * np.cos(mode * x)

    u = 0.35 * np.sin(mode * x)

    temperature = 1.0 + 0.05 * np.cos(mode * x)

    return rho, u, temperature


# ---------------------------------------------------------------------------
# Hidden kinetic perturbation
# ---------------------------------------------------------------------------


def hidden_kinetic_shape(
    v: np.ndarray,
) -> np.ndarray:
    """
    Construct a velocity-space perturbation with approximately zero
    contribution to the first three hydrodynamic moments.

    We use a Hermite-like fourth-order structure:

        H4(z) = z^4 - 6 z^2 + 3

    multiplied by a Maxwellian.

    This changes the higher-order kinetic structure while leaving rho, u and
    temperature essentially unchanged at the level of the continuous moments.
    """

    z = v / np.sqrt(2.0)

    h4 = z**4 - 6.0 * z**2 + 3.0

    weight = np.exp(-0.5 * v**2)

    shape = h4 * weight

    # Remove small numerical projection onto 1, v and v^2 using a discrete
    # weighted least-squares projection. This makes the construction more
    # robust on the finite velocity grid.
    basis = np.vstack(
        [
            np.ones_like(v),
            v,
            v**2,
        ]
    ).T

    weighted_basis = basis * weight[:, None]

    coefficients, *_ = np.linalg.lstsq(
        weighted_basis,
        shape,
        rcond=None,
    )

    correction = basis @ coefficients

    shape = shape - correction * weight

    # Normalize to unit maximum magnitude.
    maximum = np.max(np.abs(shape))

    if maximum > 0.0:
        shape /= maximum

    return shape


# ---------------------------------------------------------------------------
# Initial distributions A and B
# ---------------------------------------------------------------------------


def make_initial_distributions(
    x: np.ndarray,
    v: np.ndarray,
    L: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct two kinetic states with identical macroscopic moments.

    State A:
        local Maxwellian.

    State B:
        Maxwellian plus a higher-order velocity-space perturbation.

    The perturbation is spatially modulated by the same macroscopic scale L.
    """

    rho, u, temperature = initial_hydro_state(x, L)

    equilibrium = maxwellian(
        rho,
        u,
        temperature,
        v,
    )

    velocity_shape = hidden_kinetic_shape(v)

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

    # Prevent tiny negative values caused by the finite perturbation.
    f_b = np.maximum(f_b, 1.0e-14)

    # Rescale B so that its density remains extremely close to A.
    rho_a, _, _ = hydro_moments(f_a, v)
    rho_b, _, _ = hydro_moments(f_b, v)

    correction = (
        rho_a / np.maximum(rho_b, DENSITY_FLOOR)
    )

    f_b *= correction[:, None]

    return f_a, f_b


# ---------------------------------------------------------------------------
# Nonlocal spatial filtering
# ---------------------------------------------------------------------------


def gaussian_kernel(
    a: float,
) -> np.ndarray:
    """
    Construct a periodic discrete Gaussian kernel with width a.

    The kernel is normalized so that its discrete weights sum to one.

    This is the explicit microscopic length entering the experiment.
    """

    if a <= 0.0:
        raise ValueError("Microscopic scale a must be positive.")

    # Include enough points to capture the Gaussian tails.
    radius = max(
        1,
        int(np.ceil(4.0 * a / DX)),
    )

    offsets = np.arange(
        -radius,
        radius + 1,
        dtype=float,
    )

    distances = offsets * DX

    kernel = np.exp(
        -0.5 * (distances / a) ** 2
    )

    kernel /= np.sum(kernel)

    return kernel


def periodic_filter(
    f: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """
    Apply the spatial kernel independently at every velocity.

    Periodicity is implemented with np.roll.
    """

    filtered = np.zeros_like(f)

    centre = len(kernel) // 2

    for index, weight in enumerate(kernel):
        shift = index - centre

        filtered += weight * np.roll(
            f,
            shift=shift,
            axis=0,
        )

    return filtered


# ---------------------------------------------------------------------------
# Nonlocal BGK dynamics
# ---------------------------------------------------------------------------


def nonlocal_bgk_rhs(
    f: np.ndarray,
    v: np.ndarray,
    tau: float,
    kernel: np.ndarray,
) -> np.ndarray:
    """
    Compute the RHS of the nonlocal BGK equation.

        df/dt + v df/dx = (f_eq[filtered(f)] - f) / tau

    Spatial streaming uses a first-order upwind discretization.

    The experiment is intentionally simple and should not be interpreted as
    a high-order kinetic solver.
    """

    if tau <= 0.0:
        raise ValueError("tau must be positive.")

    # Upwind streaming.
    streaming = np.zeros_like(f)

    for j, velocity in enumerate(v):
        if velocity >= 0.0:
            derivative = (
                f[:, j]
                - np.roll(f[:, j], 1)
            ) / DX
        else:
            derivative = (
                np.roll(f[:, j], -1)
                - f[:, j]
            ) / DX

        streaming[:, j] = -velocity * derivative

    # Explicit microscopic nonlocality.
    filtered = periodic_filter(
        f,
        kernel,
    )

    rho, u, temperature = hydro_moments(
        filtered,
        v,
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

    return streaming + relaxation


# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------


def rk2_step(
    f: np.ndarray,
    v: np.ndarray,
    dt: float,
    tau: float,
    kernel: np.ndarray,
) -> np.ndarray:
    """
    Two-stage midpoint RK2 step.

    RK2 is used here instead of forward Euler because the relaxation term
    becomes stiff for the smaller tau values.
    """

    k1 = nonlocal_bgk_rhs(
        f,
        v,
        tau,
        kernel,
    )

    midpoint = f + 0.5 * dt * k1

    k2 = nonlocal_bgk_rhs(
        midpoint,
        v,
        tau,
        kernel,
    )

    result = f + dt * k2

    return np.maximum(
        result,
        1.0e-14,
    )


# ---------------------------------------------------------------------------
# Discrepancy measurement
# ---------------------------------------------------------------------------


def calculate_discrepancy(
    f_a: np.ndarray,
    f_b: np.ndarray,
    v: np.ndarray,
) -> tuple[float, float, float, float]:
    """
    Compare the hydrodynamic states generated by A and B.

    Returns
    -------
    rho_rms
    velocity_rms
    temperature_rms
    combined
    """

    rho_a, u_a, temp_a = hydro_moments(
        f_a,
        v,
    )

    rho_b, u_b, temp_b = hydro_moments(
        f_b,
        v,
    )

    rho_diff = rho_a - rho_b
    velocity_diff = u_a - u_b
    temperature_diff = temp_a - temp_b

    rho_rms = np.sqrt(
        np.mean(rho_diff**2)
    )

    velocity_rms = np.sqrt(
        np.mean(velocity_diff**2)
    )

    temperature_rms = np.sqrt(
        np.mean(temperature_diff**2)
    )

    combined = np.sqrt(
        rho_rms**2
        + velocity_rms**2
        + temperature_rms**2
    )

    return (
        rho_rms,
        velocity_rms,
        temperature_rms,
        combined,
    )


# ---------------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------------


def run_case(
    x: np.ndarray,
    v: np.ndarray,
    L: float,
    a: float,
    tau: float,
) -> dict[str, float]:
    """
    Run one A/B hidden-state experiment.
    """

    f_a, f_b = make_initial_distributions(
        x,
        v,
        L,
    )

    kernel = gaussian_kernel(a)

    initial = calculate_discrepancy(
        f_a,
        f_b,
        v,
    )

    n_steps = int(
        round(T_FINAL / DT)
    )

    maximum_combined = initial[3]
    maximum_time = 0.0

    final_values = initial

    for step in range(1, n_steps + 1):
        f_a = rk2_step(
            f_a,
            v,
            DT,
            tau,
            kernel,
        )

        f_b = rk2_step(
            f_b,
            v,
            DT,
            tau,
            kernel,
        )

        current = calculate_discrepancy(
            f_a,
            f_b,
            v,
        )

        if current[3] > maximum_combined:
            maximum_combined = current[3]
            maximum_time = step * DT

        final_values = current

    R = L / a

    return {
        "L": L,
        "a": a,
        "L_over_a": R,
        "tau": tau,
        "initial_density": initial[0],
        "initial_velocity": initial[1],
        "initial_temperature": initial[2],
        "initial_combined": initial[3],
        "final_density": final_values[0],
        "final_velocity": final_values[1],
        "final_temperature": final_values[2],
        "final_combined": final_values[3],
        "maximum_combined": maximum_combined,
        "time_of_maximum": maximum_time,
    }


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------


def write_csv(
    rows: list[dict[str, float]],
    path: Path,
) -> None:
    """Write experiment results to CSV."""

    if not rows:
        return

    fieldnames = list(rows[0].keys())

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
    """Write a human-readable summary."""

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "Experiment 005f: Explicit Microscopic Scale / Nonlocal BGK\n"
        )
        handle.write("=" * 68 + "\n\n")

        handle.write(
            "This experiment introduces an explicit microscopic spatial "
            "scale a through a nonlocal BGK relaxation operator.\n\n"
        )

        handle.write(
            "The principal scale ratio is R = L / a.\n"
        )

        handle.write(
            "Small R means the macroscopic structure approaches the "
            "microscopic interaction scale.\n\n"
        )

        handle.write(
            f"NX={NX}, NV={NV}, DT={DT}, T_FINAL={T_FINAL}\n"
        )
        handle.write(
            f"L values={list(L_VALUES)}\n"
        )
        handle.write(
            f"a values={list(A_VALUES)}\n"
        )
        handle.write(
            f"tau values={list(TAU_VALUES)}\n\n"
        )

        # Global extrema.
        final_values = np.array(
            [row["final_combined"] for row in rows]
        )

        maximum_values = np.array(
            [row["maximum_combined"] for row in rows]
        )

        handle.write(
            f"Final discrepancy range: "
            f"{final_values.min():.8f} "
            f"to {final_values.max():.8f}\n"
        )

        handle.write(
            f"Maximum discrepancy range: "
            f"{maximum_values.min():.8f} "
            f"to {maximum_values.max():.8f}\n\n"
        )

        # Results grouped by tau.
        handle.write(
            "Results grouped by tau\n"
        )
        handle.write(
            "-" * 68 + "\n"
        )

        for tau in TAU_VALUES:
            subset = [
                row
                for row in rows
                if np.isclose(row["tau"], tau)
            ]

            subset.sort(
                key=lambda row: row["L_over_a"]
            )

            handle.write(
                f"\ntau = {tau:.5f}\n"
            )

            handle.write(
                "  L/a       L         a       "
                "final_delta       max_delta\n"
            )

            for row in subset:
                handle.write(
                    f"  "
                    f"{row['L_over_a']:8.3f} "
                    f"{row['L']:8.4f} "
                    f"{row['a']:8.4f} "
                    f"{row['final_combined']:14.8f} "
                    f"{row['maximum_combined']:14.8f}\n"
                )

        # Fixed-L/a comparisons.
        handle.write(
            "\n\nEqual L/a comparisons\n"
        )
        handle.write(
            "-" * 68 + "\n"
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
                [row["final_combined"] for row in subset]
            )

            handle.write(
                f"L/a={ratio:.5f}: "
                f"min={values.min():.8f}, "
                f"max={values.max():.8f}, "
                f"spread={values.max() - values.min():.8f}\n"
            )

        handle.write(
            "\n\nInterpretation guide\n"
        )
        handle.write(
            "-" * 68 + "\n"
        )

        handle.write(
            "A strong increase in discrepancy as L/a decreases would "
            "support the existence of a scale-dependent closure boundary "
            "in this toy model.\n\n"
        )

        handle.write(
            "However, this experiment is not a derivation of "
            "Navier-Stokes and does not establish a physical mechanism "
            "for Navier-Stokes singularities.\n\n"
        )

        handle.write(
            "The microscopic scale a is introduced explicitly through "
            "the nonlocal relaxation operator. This distinguishes 005f "
            "from 005d, where the apparent scale dependence was partly "
            "confounded with relaxation time.\n"
        )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_discrepancy_vs_ratio(
    rows: list[dict[str, float]],
) -> None:
    """Plot final discrepancy against L/a."""

    plt.figure()

    for tau in TAU_VALUES:
        subset = [
            row
            for row in rows
            if np.isclose(row["tau"], tau)
        ]

        subset.sort(
            key=lambda row: row["L_over_a"]
        )

        x = np.array(
            [row["L_over_a"] for row in subset]
        )

        y = np.array(
            [row["final_combined"] for row in subset]
        )

        plt.plot(
            x,
            y,
            marker="o",
            label=f"tau={tau:g}",
        )

    plt.xscale("log")

    plt.xlabel("Macroscopic / microscopic scale L/a")
    plt.ylabel("Final hydrodynamic discrepancy")
    plt.title(
        "005f: Closure discrepancy versus explicit scale ratio"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "005f_discrepancy_vs_L_over_a.png",
        dpi=160,
    )

    plt.close()


def plot_discrepancy_vs_microscopic_scale(
    rows: list[dict[str, float]],
) -> None:
    """Plot discrepancy against a for fixed L and tau."""

    plt.figure()

    selected_tau = TAU_VALUES[1]

    for L in L_VALUES:
        subset = [
            row
            for row in rows
            if np.isclose(row["L"], L)
            and np.isclose(row["tau"], selected_tau)
        ]

        subset.sort(
            key=lambda row: row["a"]
        )

        x = np.array(
            [row["a"] for row in subset]
        )

        y = np.array(
            [row["final_combined"] for row in subset]
        )

        plt.plot(
            x,
            y,
            marker="o",
            label=f"L={L:g}",
        )

    plt.xlabel("Microscopic scale a")
    plt.ylabel("Final hydrodynamic discrepancy")
    plt.title(
        f"005f: Scale effect at tau={selected_tau:g}"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "005f_discrepancy_vs_microscopic_scale.png",
        dpi=160,
    )

    plt.close()


def plot_parameter_map(
    rows: list[dict[str, float]],
) -> None:
    """Create a parameter map of L/a versus tau."""

    plt.figure()

    x = np.array(
        [row["L_over_a"] for row in rows]
    )

    y = np.array(
        [row["tau"] for row in rows]
    )

    z = np.array(
        [row["final_combined"] for row in rows]
    )

    scatter = plt.scatter(
        x,
        y,
        c=z,
        s=70,
    )

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Macroscopic / microscopic scale L/a")
    plt.ylabel("Relaxation time tau")
    plt.title(
        "005f: Closure discrepancy parameter map"
    )

    colorbar = plt.colorbar(scatter)
    colorbar.set_label(
        "Final hydrodynamic discrepancy"
    )

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "005f_parameter_map.png",
        dpi=160,
    )

    plt.close()


def plot_discrepancy_vs_tau(
    rows: list[dict[str, float]],
) -> None:
    """Plot final discrepancy against tau for fixed L/a."""

    plt.figure()

    # Choose several representative microscopic scales.
    selected_a = A_VALUES[1]

    for L in L_VALUES:
        subset = [
            row
            for row in rows
            if np.isclose(row["L"], L)
            and np.isclose(row["a"], selected_a)
        ]

        subset.sort(
            key=lambda row: row["tau"]
        )

        x = np.array(
            [row["tau"] for row in subset]
        )

        y = np.array(
            [row["final_combined"] for row in subset]
        )

        plt.plot(
            x,
            y,
            marker="o",
            label=f"L={L:g}",
        )

    plt.xscale("log")

    plt.xlabel("Relaxation time tau")
    plt.ylabel("Final hydrodynamic discrepancy")
    plt.title(
        f"005f: Relaxation dependence at a={selected_a:g}"
    )

    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR
        / "005f_discrepancy_vs_tau.png",
        dpi=160,
    )

    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the complete 005f experiment."""

    print("=" * 68)
    print("Experiment 005f: Explicit Microscopic Scale / Nonlocal BGK")
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

                rows.append(result)

                print(
                    f"    initial={result['initial_combined']:.8e}"
                )

                print(
                    f"    final={result['final_combined']:.8e}"
                )

                print(
                    f"    maximum={result['maximum_combined']:.8e}"
                )

    csv_path = (
        RESULTS_DIR
        / "005f_nonlocal_microscopic_scale.csv"
    )

    summary_path = (
        RESULTS_DIR
        / "005f_nonlocal_microscopic_scale_summary.txt"
    )

    write_csv(
        rows,
        csv_path,
    )

    write_summary(
        rows,
        summary_path,
    )

    plot_discrepancy_vs_ratio(
        rows
    )

    plot_discrepancy_vs_microscopic_scale(
        rows
    )

    plot_parameter_map(
        rows
    )

    plot_discrepancy_vs_tau(
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
        "  results/005f_discrepancy_vs_L_over_a.png"
    )
    print(
        "  results/005f_discrepancy_vs_microscopic_scale.png"
    )
    print(
        "  results/005f_parameter_map.png"
    )
    print(
        "  results/005f_discrepancy_vs_tau.png"
    )


if __name__ == "__main__":
    main()
