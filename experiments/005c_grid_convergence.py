"""
Experiment 005c: Grid Convergence

Purpose
-------
Determine whether the scale dependence observed in Experiment 005b
survives spatial grid refinement.

We repeat the controlled scale-separation experiment at several
spatial resolutions while keeping the physical parameters fixed:

    L = 2*pi / k
    tau = epsilon * L / v_ref

The main question is:

    Does the measured kinetic -> hydrodynamic closure error converge
    as the spatial grid is refined?

If results converge, the observed scale dependence is less likely to
be a simple spatial-discretization artifact.

If results change substantially with grid resolution, numerical
resolution is an important confounding factor.

This remains a controlled BGK toy model, not a physical Navier-Stokes
calculation and not a measurement of a physical Knudsen number.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

NX_VALUES = [128, 256, 512]
NV = 96

DOMAIN_LENGTH = 2.0 * np.pi
X_MIN = 0.0
X_MAX = DOMAIN_LENGTH

V_MIN = -4.0
V_MAX = 4.0

DT = 0.001
T_FINAL = 4.0

REFERENCE_SPEED = 1.0

SPATIAL_MODES = [1, 4, 8, 16, 24]
EPSILON_TARGETS = [0.01, 0.05, 0.10]

HIDDEN_AMPLITUDE = 0.35


# ---------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------

def maxwellian(v, density, velocity, temperature):
    """Return a one-dimensional Maxwellian."""
    temperature = np.maximum(
        np.asarray(temperature),
        1e-12,
    )

    return (
        density
        / np.sqrt(2.0 * np.pi * temperature)
        * np.exp(
            -0.5 * (v - velocity) ** 2 / temperature
        )
    )


def velocity_grid():
    """Return velocity grid and spacing."""
    v = np.linspace(V_MIN, V_MAX, NV, endpoint=False)
    dv = (V_MAX - V_MIN) / NV
    return v, dv


def initial_distributions(nx):
    """
    Construct two kinetic states.

    State A is Maxwellian.

    State B contains a higher-order velocity perturbation but is
    corrected so that density, momentum, and energy match A exactly
    at every spatial location.
    """
    x = np.linspace(X_MIN, X_MAX, nx, endpoint=False)
    dx = DOMAIN_LENGTH / nx

    v, dv = velocity_grid()

    X, V = np.meshgrid(x, v, indexing="ij")

    density = np.ones_like(X)
    velocity = 0.35 * np.sin(2.0 * X)
    temperature = np.ones_like(X)

    f_a = maxwellian(V, density, velocity, temperature)

    # Dimensionless velocity relative to the local Maxwellian.
    c = (V - velocity) / np.sqrt(temperature)

    # Fourth-order Hermite-like perturbation.
    hidden = (
        HIDDEN_AMPLITUDE
        * np.cos(2.0 * X)
        * (c**4 - 6.0 * c**2 + 3.0)
        * f_a
    )

    f_b = f_a + hidden

    # Match density, momentum and energy of B to A.
    f_b = match_hydrodynamic_moments(
        f_a,
        f_b,
        V,
        dv,
    )

    return x, v, dx, dv, f_a, f_b

def match_hydrodynamic_moments(f_reference, f_target, v, dv):
    """
    Correct f_target so that its density, momentum, and second velocity
    moment match f_reference.

    The velocity coordinate is scaled to [-1, 1] before constructing
    the moment system. This greatly improves numerical conditioning.
    """

    # Scale velocity to approximately [-1, 1].
    velocity_scale = max(
        float(np.max(np.abs(v))),
        1.0,
    )

    s = v / velocity_scale

    # Basis functions for the correction.
    basis = np.vstack(
        [
            np.ones_like(s),
            s,
            s**2,
        ]
    )

    # Construct the 3x3 moment matrix.
    moment_matrix = (
        basis * dv
    ) @ basis.T

    corrected = np.empty_like(f_target)

    for i in range(f_target.shape[0]):

        difference = (
            f_reference[i] - f_target[i]
        )

        # Moments of the difference.
        rhs = np.array(
            [
                np.sum(dv * difference),
                np.sum(dv * s * difference),
                np.sum(dv * s**2 * difference),
            ],
            dtype=float,
        )

        coefficients = np.linalg.solve(
            moment_matrix,
            rhs,
        )

        correction = (
            coefficients[0]
            + coefficients[1] * s
            + coefficients[2] * s**2
        )

        corrected[i] = (
            f_target[i] + correction
        )

    return corrected

# ---------------------------------------------------------------------
# Hydrodynamic moments
# ---------------------------------------------------------------------

def hydrodynamic_moments(f, v, dv):
    """
    Compute density, mean velocity and temperature-like second moment.
    """
    density = np.sum(f, axis=1) * dv

    momentum = np.sum(f * v[None, :], axis=1) * dv

    velocity = momentum / np.maximum(density, 1e-14)

    energy = np.sum(
        f * v[None, :] ** 2,
        axis=1,
    ) * dv

    temperature = (
        energy / np.maximum(density, 1e-14)
        - velocity**2
    )

    return density, velocity, temperature


def hydrodynamic_difference(
    f_a,
    f_b,
    v,
    dv,
):
    """
    Return RMS differences in density, velocity and temperature.
    """
    rho_a, u_a, t_a = hydrodynamic_moments(f_a, v, dv)
    rho_b, u_b, t_b = hydrodynamic_moments(f_b, v, dv)

    density_difference = np.sqrt(
        np.mean((rho_a - rho_b) ** 2)
    )

    velocity_difference = np.sqrt(
        np.mean((u_a - u_b) ** 2)
    )

    temperature_difference = np.sqrt(
        np.mean((t_a - t_b) ** 2)
    )

    combined = np.sqrt(
        density_difference**2
        + velocity_difference**2
        + temperature_difference**2
    )

    return (
        density_difference,
        velocity_difference,
        temperature_difference,
        combined,
    )


def kinetic_difference(f_a, f_b, dx, dv):
    """Return the L2 difference between two kinetic states."""
    return np.sqrt(
        np.sum((f_a - f_b) ** 2) * dx * dv
    )


# ---------------------------------------------------------------------
# Kinetic model
# ---------------------------------------------------------------------

def equilibrium(f, v, dv):
    """Construct local Maxwellian equilibrium from moments."""
    density, velocity, temperature = hydrodynamic_moments(
        f,
        v,
        dv,
    )

    return maxwellian(
        v[None, :],
        density[:, None],
        velocity[:, None],
        temperature[:, None],
    )


def stream_upwind(f, v, dx, dt):
    """
    First-order periodic upwind streaming.

    Solves approximately:

        df/dt + v df/dx = 0
    """
    result = np.empty_like(f)

    for j, velocity in enumerate(v):
        cfl = velocity * dt / dx

        if velocity >= 0.0:
            result[:, j] = (
                f[:, j]
                - cfl * (f[:, j] - np.roll(f[:, j], 1))
            )
        else:
            result[:, j] = (
                f[:, j]
                - cfl * (np.roll(f[:, j], -1) - f[:, j])
            )

    return result


def bgk_step(f, v, dx, dv, dt, tau):
    """
    Advance one BGK step.

    Streaming is first-order upwind.
    Relaxation is applied exactly through exponential decay.
    """
    streamed = stream_upwind(
        f,
        v,
        dx,
        dt,
    )

    f_eq = equilibrium(
        streamed,
        v,
        dv,
    )

    relaxation_factor = np.exp(-dt / tau)

    return (
        relaxation_factor * streamed
        + (1.0 - relaxation_factor) * f_eq
    )


def run_case(nx, spatial_mode, epsilon):
    """
    Run one grid-convergence case.

    Returns maximum and final hydrodynamic and kinetic differences.
    """
    (
        x,
        v,
        dx,
        dv,
        f_a,
        f_b,
    ) = initial_distributions(nx)

    L = DOMAIN_LENGTH / spatial_mode

    tau = (
        epsilon
        * L
        / REFERENCE_SPEED
    )

    n_steps = int(round(T_FINAL / DT))

    initial_hydro = hydrodynamic_difference(
        f_a,
        f_b,
        v,
        dv,
    )

    initial_kinetic = kinetic_difference(
        f_a,
        f_b,
        dx,
        dv,
    )

    maximum_density = 0.0
    maximum_velocity = 0.0
    maximum_temperature = 0.0
    maximum_combined = 0.0
    maximum_kinetic = initial_kinetic

    final_density = 0.0
    final_velocity = 0.0
    final_temperature = 0.0
    final_combined = 0.0
    final_kinetic = initial_kinetic

    for _ in range(n_steps):
        f_a = bgk_step(
            f_a,
            v,
            dx,
            dv,
            DT,
            tau,
        )

        f_b = bgk_step(
            f_b,
            v,
            dx,
            dv,
            DT,
            tau,
        )

        (
            density_difference,
            velocity_difference,
            temperature_difference,
            combined,
        ) = hydrodynamic_difference(
            f_a,
            f_b,
            v,
            dv,
        )

        kinetic = kinetic_difference(
            f_a,
            f_b,
            dx,
            dv,
        )

        maximum_density = max(
            maximum_density,
            density_difference,
        )

        maximum_velocity = max(
            maximum_velocity,
            velocity_difference,
        )

        maximum_temperature = max(
            maximum_temperature,
            temperature_difference,
        )

        maximum_combined = max(
            maximum_combined,
            combined,
        )

        maximum_kinetic = max(
            maximum_kinetic,
            kinetic,
        )

        final_density = density_difference
        final_velocity = velocity_difference
        final_temperature = temperature_difference
        final_combined = combined
        final_kinetic = kinetic

    return {
        "nx": nx,
        "epsilon_target": epsilon,
        "epsilon_actual": tau * REFERENCE_SPEED / L,
        "spatial_mode": spatial_mode,
        "L": L,
        "tau": tau,
        "dx": dx,
        "cells_per_wavelength": L / dx,
        "initial_hydro": np.sqrt(
            sum(value**2 for value in initial_hydro)
        ),
        "initial_kinetic": initial_kinetic,
        "maximum_density": maximum_density,
        "final_density": final_density,
        "maximum_velocity": maximum_velocity,
        "final_velocity": final_velocity,
        "maximum_temperature": maximum_temperature,
        "final_temperature": final_temperature,
        "maximum_combined": maximum_combined,
        "final_combined": final_combined,
        "maximum_kinetic": maximum_kinetic,
        "final_kinetic": final_kinetic,
    }


# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------

def run_experiment():
    results = []

    for nx in NX_VALUES:
        for epsilon in EPSILON_TARGETS:
            for spatial_mode in SPATIAL_MODES:
                print(
                    f"Running NX={nx}, "
                    f"epsilon={epsilon:.3f}, "
                    f"k={spatial_mode}"
                )

                result = run_case(
                    nx,
                    spatial_mode,
                    epsilon,
                )

                results.append(result)

    return results


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def save_outputs(results):
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # CSV
    csv_path = (
        results_dir
        / "experiment_005c_summary.csv"
    )

    fieldnames = [
        "nx",
        "epsilon_target",
        "epsilon_actual",
        "spatial_mode",
        "L",
        "tau",
        "cells_per_wavelength",
        "initial_hydro",
        "initial_kinetic",
        "maximum_density",
        "final_density",
        "maximum_velocity",
        "final_velocity",
        "maximum_temperature",
        "final_temperature",
        "maximum_combined",
        "final_combined",
        "maximum_kinetic",
        "final_kinetic",
    ]

    import csv

    with csv_path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(result)

    # Summary text
    summary_path = (
        results_dir
        / "experiment_005c_summary.txt"
    )

    with summary_path.open("w") as handle:
        handle.write(
            "Experiment 005c: Grid Convergence\n"
        )
        handle.write(
            "=================================\n\n"
        )

        handle.write(
            f"Spatial grids: "
            f"{', '.join(str(n) for n in NX_VALUES)}\n"
        )

        handle.write(
            f"Velocity grid: {NV}\n"
        )

        handle.write(
            f"Time step: {DT}\n"
        )

        handle.write(
            f"Final time: {T_FINAL}\n"
        )

        handle.write(
            f"Epsilon targets: "
            f"{', '.join(f'{e:.4f}' for e in EPSILON_TARGETS)}\n\n"
        )

        handle.write("Results\n")
        handle.write("-------\n")

        handle.write(
            "nx,epsilon,spatial_mode,L,"
            "cells_per_wavelength,"
            "maximum_combined,final_combined,"
            "maximum_kinetic,final_kinetic\n"
        )

        for result in results:
            handle.write(
                f"{result['nx']},"
                f"{result['epsilon_target']:.6e},"
                f"{result['spatial_mode']},"
                f"{result['L']:.12e},"
                f"{result['cells_per_wavelength']:.6f},"
                f"{result['maximum_combined']:.12e},"
                f"{result['final_combined']:.12e},"
                f"{result['maximum_kinetic']:.12e},"
                f"{result['final_kinetic']:.12e}\n"
            )

    # -------------------------------------------------------------
    # Plot 1: final discrepancy versus wavelength
    # -------------------------------------------------------------

    for epsilon in EPSILON_TARGETS:
        plt.figure()

        for nx in NX_VALUES:
            subset = [
                r for r in results
                if (
                    r["epsilon_target"] == epsilon
                    and r["nx"] == nx
                )
            ]

            subset.sort(
                key=lambda r: r["L"]
            )

            plt.plot(
                [r["L"] for r in subset],
                [r["final_combined"] for r in subset],
                marker="o",
                label=f"NX={nx}",
            )

        plt.xlabel("Wavelength L")
        plt.ylabel(
            "Final hydrodynamic difference"
        )

        plt.title(
            f"Experiment 005c: Grid convergence "
            f"(epsilon={epsilon:.2f})"
        )

        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            results_dir
            / f"experiment_005c_final_epsilon_{epsilon:.2f}.png",
            dpi=150,
        )

        plt.close()

    # -------------------------------------------------------------
    # Plot 2: maximum discrepancy versus wavelength
    # -------------------------------------------------------------

    for epsilon in EPSILON_TARGETS:
        plt.figure()

        for nx in NX_VALUES:
            subset = [
                r for r in results
                if (
                    r["epsilon_target"] == epsilon
                    and r["nx"] == nx
                )
            ]

            subset.sort(
                key=lambda r: r["L"]
            )

            plt.plot(
                [r["L"] for r in subset],
                [r["maximum_combined"] for r in subset],
                marker="o",
                label=f"NX={nx}",
            )

        plt.xlabel("Wavelength L")
        plt.ylabel(
            "Maximum hydrodynamic difference"
        )

        plt.title(
            f"Experiment 005c: Maximum discrepancy "
            f"(epsilon={epsilon:.2f})"
        )

        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            results_dir
            / f"experiment_005c_maximum_epsilon_{epsilon:.2f}.png",
            dpi=150,
        )

        plt.close()

    # -------------------------------------------------------------
    # Plot 3: convergence versus cells per wavelength
    # -------------------------------------------------------------

    for epsilon in EPSILON_TARGETS:
        plt.figure()

        for spatial_mode in SPATIAL_MODES:
            subset = [
                r for r in results
                if (
                    r["epsilon_target"] == epsilon
                    and r["spatial_mode"] == spatial_mode
                )
            ]

            subset.sort(
                key=lambda r: r["cells_per_wavelength"]
            )

            plt.plot(
                [r["cells_per_wavelength"] for r in subset],
                [r["final_combined"] for r in subset],
                marker="o",
                label=f"k={spatial_mode}",
            )

        plt.xlabel(
            "Cells per wavelength"
        )

        plt.ylabel(
            "Final hydrodynamic difference"
        )

        plt.title(
            f"Experiment 005c: Resolution convergence "
            f"(epsilon={epsilon:.2f})"
        )

        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        plt.savefig(
            results_dir
            / f"experiment_005c_resolution_epsilon_{epsilon:.2f}.png",
            dpi=150,
        )

        plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Allow direct execution from repository root.
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    results = run_experiment()
    save_outputs(results)

    print()
    print(
        "Experiment 005c complete."
    )
    print(
        "Results written to results/"
    )
