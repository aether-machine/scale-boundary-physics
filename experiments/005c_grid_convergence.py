"""
Experiment 005c: Grid convergence at controlled scale separation.

Purpose
-------
Test whether the kinetic-to-hydrodynamic discrepancy observed in
Experiment 005b survives spatial grid refinement.

The experiment varies:

    NX              spatial resolution
    spatial_mode    physical perturbation wavenumber k
    epsilon         relaxation-scale ratio

with

    L   = DOMAIN_LENGTH / spatial_mode
    tau = epsilon * L / REFERENCE_SPEED

The spatial mode is part of the actual initial condition. Therefore,
unlike the earlier version of this experiment, changing spatial_mode
really changes the physical wavelength being represented.

The experiment compares two kinetic states which have identical
hydrodynamic moments initially but differ in higher-order velocity-space
structure.

The central diagnostic is whether that hidden kinetic difference produces
different subsequent hydrodynamic evolution.

This is a controlled numerical experiment in a toy BGK kinetic model.
It is NOT a derivation of Navier-Stokes and should not be interpreted
as one.
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


# =====================================================================
# Configuration
# =====================================================================

NX_VALUES = [128, 256, 512]

NV = 96

DOMAIN_LENGTH = 2.0 * np.pi

V_MIN = -4.0
V_MAX = 4.0

DT = 0.001
T_FINAL = 4.0

REFERENCE_SPEED = 1.0

SPATIAL_MODES = [1, 4, 8, 16, 24]

EPSILON_TARGETS = [0.01, 0.05, 0.10]

HIDDEN_AMPLITUDE = 0.35

OUTPUT_DIR = Path(
    "results/005c_grid_convergence"
)


# =====================================================================
# Grids
# =====================================================================

def velocity_grid():
    """
    Return the one-dimensional velocity grid and velocity spacing.
    """

    dv = (
        V_MAX - V_MIN
    ) / NV

    v = (
        V_MIN
        + dv * np.arange(NV)
    )

    return v, dv


def spatial_grid(nx):
    """
    Return the periodic spatial grid and spatial spacing.
    """

    dx = (
        DOMAIN_LENGTH / nx
    )

    x = dx * np.arange(nx)

    return x, dx


# =====================================================================
# Maxwellian
# =====================================================================

def maxwellian(
    density,
    velocity,
    temperature,
    v,
):
    """
    Construct a local Maxwellian.

    Parameters
    ----------
    density:
        Shape (NX,)

    velocity:
        Shape (NX,)

    temperature:
        Shape (NX,)

    v:
        Shape (NV,)

    Returns
    -------
    f:
        Shape (NX, NV)
    """

    density = np.asarray(
        density,
        dtype=float,
    )

    velocity = np.asarray(
        velocity,
        dtype=float,
    )

    temperature = np.asarray(
        temperature,
        dtype=float,
    )

    v = np.asarray(
        v,
        dtype=float,
    )

    if density.ndim != 1:
        raise ValueError(
            "density must be 1-D. "
            f"Received shape {density.shape}."
        )

    if velocity.ndim != 1:
        raise ValueError(
            "velocity must be 1-D. "
            f"Received shape {velocity.shape}."
        )

    if temperature.ndim != 1:
        raise ValueError(
            "temperature must be 1-D. "
            f"Received shape {temperature.shape}."
        )

    if v.ndim != 1:
        raise ValueError(
            "v must be 1-D. "
            f"Received shape {v.shape}."
        )

    if not (
        len(density)
        == len(velocity)
        == len(temperature)
    ):
        raise ValueError(
            "Macroscopic fields must have "
            "the same spatial length."
        )

    temperature = np.maximum(
        temperature,
        1.0e-10,
    )

    velocity_difference = (
        v[None, :]
        - velocity[:, None]
    )

    prefactor = (
        density[:, None]
        / np.sqrt(
            2.0
            * np.pi
            * temperature[:, None]
        )
    )

    exponent = -(
        velocity_difference ** 2
    ) / (
        2.0
        * temperature[:, None]
    )

    f = (
        prefactor
        * np.exp(exponent)
    )

    return f


# =====================================================================
# Hydrodynamic moments
# =====================================================================

def hydrodynamic_moments(
    f,
    v,
    dv,
):
    """
    Calculate density, velocity and temperature.

    rho       = integral f dv

    rho * u   = integral v f dv

    rho(u²+T) = integral v² f dv
    """

    f = np.asarray(
        f,
        dtype=float,
    )

    v = np.asarray(
        v,
        dtype=float,
    )

    if f.ndim != 2:
        raise ValueError(
            "f must be 2-D with shape "
            "(NX, NV). "
            f"Received {f.shape}."
        )

    if v.ndim != 1:
        raise ValueError(
            "v must be 1-D."
        )

    if f.shape[1] != len(v):
        raise ValueError(
            "Velocity dimension mismatch: "
            f"f has {f.shape[1]} columns, "
            f"but v has {len(v)} points."
        )

    density = np.sum(
        f * dv,
        axis=1,
    )

    density_safe = np.maximum(
        density,
        1.0e-12,
    )

    momentum = np.sum(
        f
        * v[None, :]
        * dv,
        axis=1,
    )

    velocity = (
        momentum
        / density_safe
    )

    second_moment = np.sum(
        f
        * v[None, :] ** 2
        * dv,
        axis=1,
    )

    temperature = (
        second_moment
        / density_safe
        - velocity ** 2
    )

    temperature = np.maximum(
        temperature,
        1.0e-10,
    )

    return (
        density,
        velocity,
        temperature,
    )


def hydrodynamic_difference(
    f_a,
    f_b,
    v,
    dv,
):
    """
    Calculate RMS differences between
    the hydrodynamic moments of two states.
    """

    (
        rho_a,
        u_a,
        T_a,
    ) = hydrodynamic_moments(
        f_a,
        v,
        dv,
    )

    (
        rho_b,
        u_b,
        T_b,
    ) = hydrodynamic_moments(
        f_b,
        v,
        dv,
    )

    density_difference = np.sqrt(
        np.mean(
            (
                rho_a
                - rho_b
            ) ** 2
        )
    )

    velocity_difference = np.sqrt(
        np.mean(
            (
                u_a
                - u_b
            ) ** 2
        )
    )

    temperature_difference = np.sqrt(
        np.mean(
            (
                T_a
                - T_b
            ) ** 2
        )
    )

    combined_difference = np.sqrt(
        density_difference ** 2
        + velocity_difference ** 2
        + temperature_difference ** 2
    )

    return (
        density_difference,
        velocity_difference,
        temperature_difference,
        combined_difference,
    )


def kinetic_difference(
    f_a,
    f_b,
    dv,
):
    """
    L2 difference between two kinetic distributions.
    """

    difference = (
        f_a
        - f_b
    )

    return np.sqrt(
        np.sum(
            difference ** 2
            * dv
        )
    )


# =====================================================================
# Moment matching
# =====================================================================

def match_hydrodynamic_moments(
    f_reference,
    f_target,
    v,
    dv,
):
    """
    Modify f_target so that its density, momentum and second velocity
    moment match f_reference at every spatial point.

    The correction is constructed from three basis functions:

        1
        s
        s²

    where

        s = v / velocity_scale.

    Scaling the velocity coordinate improves conditioning but does not
    change the span of the correction basis.
    """

    if f_reference.shape != f_target.shape:
        raise ValueError(
            "Reference and target distributions "
            "must have identical shapes."
        )

    if f_reference.ndim != 2:
        raise ValueError(
            "Distributions must be 2-D."
        )

    if v.ndim != 1:
        raise ValueError(
            "v must be 1-D."
        )

    if f_reference.shape[1] != len(v):
        raise ValueError(
            "Distribution velocity dimension "
            "does not match v."
        )

    velocity_scale = max(
        float(
            np.max(
                np.abs(v)
            )
        ),
        1.0,
    )

    s = (
        v
        / velocity_scale
    )

    basis = np.vstack(
        [
            np.ones_like(s),
            s,
            s ** 2,
        ]
    )

    moment_matrix = (
        basis * dv
    ) @ basis.T

    determinant = np.linalg.det(
        moment_matrix
    )

    if not np.isfinite(
        determinant
    ):
        raise ValueError(
            "Moment matrix is not finite."
        )

    if abs(determinant) < 1.0e-14:
        raise ValueError(
            "Moment matrix is singular "
            "or nearly singular."
        )

    corrected = np.empty_like(
        f_target
    )

    for i in range(
        f_target.shape[0]
    ):

        difference = (
            f_reference[i]
            - f_target[i]
        )

        rhs = np.array(
            [
                np.sum(
                    dv
                    * difference
                ),
                np.sum(
                    dv
                    * s
                    * difference
                ),
                np.sum(
                    dv
                    * s ** 2
                    * difference
                ),
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
            + coefficients[2] * s ** 2
        )

        corrected[i] = (
            f_target[i]
            + correction
        )

    return corrected


# =====================================================================
# Initial states
# =====================================================================

def initial_distributions(
    nx,
    spatial_mode,
):
    """
    Construct the pair of microscopic states used in the experiment.

    State A:
        local Maxwellian.

    State B:
        same hydrodynamic moments as A, but with an additional
        higher-order velocity-space perturbation.

    The perturbation has physical spatial wavelength

        L = DOMAIN_LENGTH / spatial_mode.
    """

    x, dx = spatial_grid(
        nx
    )

    v, dv = velocity_grid()

    # --------------------------------------------------------------
    # Macroscopic fields.
    #
    # IMPORTANT:
    # These are functions of x only.
    # Each therefore has shape (NX,).
    # --------------------------------------------------------------

    density = (
        1.0
        + 0.05
        * np.cos(
            spatial_mode * x
        )
    )

    velocity = (
        0.35
        * np.sin(
            spatial_mode * x
        )
    )

    temperature = (
        1.0
        + 0.05
        * np.cos(
            spatial_mode * x
        )
    )

    # --------------------------------------------------------------
    # Maxwellian state A.
    #
    # maxwellian() performs the required broadcasting internally,
    # producing shape (NX, NV).
    # --------------------------------------------------------------

    f_a = maxwellian(
        density,
        velocity,
        temperature,
        v,
    )

    # --------------------------------------------------------------
    # Hidden velocity-space structure.
    # --------------------------------------------------------------

    velocity_scale = max(
        float(
            np.max(
                np.abs(v)
            )
        ),
        1.0,
    )

    c = (
        v
        / velocity_scale
    )

    velocity_shape = (
        c ** 4
        - 6.0 * c ** 2
        + 3.0
    )

    spatial_shape = (
        np.cos(
            spatial_mode * x
        )[:, None]
    )

    hidden = (
        HIDDEN_AMPLITUDE
        * spatial_shape
        * velocity_shape[None, :]
        * f_a
    )

    f_b = (
        f_a
        + hidden
    )

    # --------------------------------------------------------------
    # Moment matching.
    #
    # v is deliberately one-dimensional here.
    # --------------------------------------------------------------

    f_b = match_hydrodynamic_moments(
        f_a,
        f_b,
        v,
        dv,
    )

    # --------------------------------------------------------------
    # Strong shape checks.
    # --------------------------------------------------------------

    expected_shape = (
        nx,
        NV,
    )

    if f_a.shape != expected_shape:
        raise ValueError(
            "Initial f_a has incorrect shape: "
            f"{f_a.shape}; "
            f"expected {expected_shape}."
        )

    if f_b.shape != expected_shape:
        raise ValueError(
            "Initial f_b has incorrect shape: "
            f"{f_b.shape}; "
            f"expected {expected_shape}."
        )

    # --------------------------------------------------------------
    # Verify the defining property of the experiment:
    # the two states should have matching hydrodynamic moments.
    # --------------------------------------------------------------

    (
        initial_density_difference,
        initial_velocity_difference,
        initial_temperature_difference,
        initial_combined_difference,
    ) = hydrodynamic_difference(
        f_a,
        f_b,
        v,
        dv,
    )

    tolerance = 1.0e-10

    if (
        initial_density_difference
        > tolerance
        or initial_velocity_difference
        > tolerance
        or initial_temperature_difference
        > tolerance
    ):
        raise ValueError(
            "Moment matching failed.\n"
            f"density difference = "
            f"{initial_density_difference:.6e}\n"
            f"velocity difference = "
            f"{initial_velocity_difference:.6e}\n"
            f"temperature difference = "
            f"{initial_temperature_difference:.6e}"
        )

    return (
        x,
        dx,
        v,
        dv,
        f_a,
        f_b,
    )


# =====================================================================
# Equilibrium
# =====================================================================

def equilibrium(
    f,
    v,
    dv,
):
    """
    Construct the local Maxwellian corresponding to f.
    """

    (
        density,
        velocity,
        temperature,
    ) = hydrodynamic_moments(
        f,
        v,
        dv,
    )

    return maxwellian(
        density,
        velocity,
        temperature,
        v,
    )


# =====================================================================
# Streaming
# =====================================================================

def stream_upwind(
    f,
    v,
    dx,
    dt,
):
    """
    First-order upwind spatial streaming.

    Periodic boundary conditions are used in x.
    """

    if f.ndim != 2:
        raise ValueError(
            "stream_upwind expects f with "
            "shape (NX, NV). "
            f"Received {f.shape}."
        )

    if v.ndim != 1:
        raise ValueError(
            "stream_upwind expects v to be "
            "one-dimensional."
        )

    if f.shape[1] != len(v):
        raise ValueError(
            "f and v have incompatible "
            "velocity dimensions."
        )

    streamed = np.empty_like(
        f
    )

    for j, velocity in enumerate(v):

        column = f[:, j]

        if velocity > 0.0:

            streamed[:, j] = (
                column
                - (
                    velocity
                    * dt
                    / dx
                )
                * (
                    column
                    - np.roll(
                        column,
                        1,
                    )
                )
            )

        elif velocity < 0.0:

            streamed[:, j] = (
                column
                - (
                    velocity
                    * dt
                    / dx
                )
                * (
                    np.roll(
                        column,
                        -1,
                    )
                    - column
                )
            )

        else:

            streamed[:, j] = (
                column
            )

    return streamed


# =====================================================================
# BGK step
# =====================================================================

def bgk_step(
    f,
    v,
    dv,
    dx,
    dt,
    tau,
):
    """
    Advance one BGK timestep.

    Operator splitting:

        1. spatial streaming
        2. local BGK relaxation
    """

    if tau <= 0.0:
        raise ValueError(
            "tau must be positive."
        )

    f_streamed = stream_upwind(
        f,
        v,
        dx,
        dt,
    )

    f_eq = equilibrium(
        f_streamed,
        v,
        dv,
    )

    relaxation_factor = (
        dt / tau
    )

    f_new = (
        f_streamed
        + relaxation_factor
        * (
            f_eq
            - f_streamed
        )
    )

    # Small negative values can arise from the first-order numerical
    # scheme. Clip them to preserve a non-negative distribution.
    f_new = np.maximum(
        f_new,
        0.0,
    )

    return f_new


# =====================================================================
# One experimental case
# =====================================================================

def run_case(
    nx,
    spatial_mode,
    epsilon,
):
    """
    Run one grid / spatial-scale / epsilon combination.
    """

    (
        x,
        dx,
        v,
        dv,
        f_a,
        f_b,
    ) = initial_distributions(
        nx,
        spatial_mode,
    )

    wavelength = (
        DOMAIN_LENGTH
        / spatial_mode
    )

    L = wavelength

    tau = (
        epsilon
        * L
        / REFERENCE_SPEED
    )

    n_steps = int(
        round(
            T_FINAL / DT
        )
    )

    # --------------------------------------------------------------
    # Initial diagnostics
    # --------------------------------------------------------------

    initial_kinetic = (
        kinetic_difference(
            f_a,
            f_b,
            dv,
        )
    )

    (
        initial_density,
        initial_velocity,
        initial_temperature,
        initial_combined,
    ) = hydrodynamic_difference(
        f_a,
        f_b,
        v,
        dv,
    )

    max_density = (
        initial_density
    )

    max_velocity = (
        initial_velocity
    )

    max_temperature = (
        initial_temperature
    )

    max_combined = (
        initial_combined
    )

    # --------------------------------------------------------------
    # Time integration
    # --------------------------------------------------------------

    for step in range(
        n_steps
    ):

        f_a = bgk_step(
            f_a,
            v,
            dv,
            dx,
            DT,
            tau,
        )

        f_b = bgk_step(
            f_b,
            v,
            dv,
            dx,
            DT,
            tau,
        )

        (
            density_difference,
            velocity_difference,
            temperature_difference,
            combined_difference,
        ) = hydrodynamic_difference(
            f_a,
            f_b,
            v,
            dv,
        )

        max_density = max(
            max_density,
            density_difference,
        )

        max_velocity = max(
            max_velocity,
            velocity_difference,
        )

        max_temperature = max(
            max_temperature,
            temperature_difference,
        )

        max_combined = max(
            max_combined,
            combined_difference,
        )

    # --------------------------------------------------------------
    # Final diagnostics
    # --------------------------------------------------------------

    (
        final_density,
        final_velocity,
        final_temperature,
        final_combined,
    ) = hydrodynamic_difference(
        f_a,
        f_b,
        v,
        dv,
    )

    final_kinetic = (
        kinetic_difference(
            f_a,
            f_b,
            dv,
        )
    )

    epsilon_actual = (
        tau
        * REFERENCE_SPEED
        / L
    )

    cells_per_wavelength = (
        wavelength / dx
    )

    return {
        "nx": nx,
        "spatial_mode": spatial_mode,
        "wavelength": wavelength,
        "L": L,
        "epsilon_target": epsilon,
        "epsilon_actual": epsilon_actual,
        "tau": tau,
        "dx": dx,
        "cells_per_wavelength": (
            cells_per_wavelength
        ),
        "initial_kinetic": (
            initial_kinetic
        ),
        "initial_hydro": (
            initial_combined
        ),
        "max_density": (
            max_density
        ),
        "max_velocity": (
            max_velocity
        ),
        "max_temperature": (
            max_temperature
        ),
        "max_combined": (
            max_combined
        ),
        "final_density": (
            final_density
        ),
        "final_velocity": (
            final_velocity
        ),
        "final_temperature": (
            final_temperature
        ),
        "final_combined": (
            final_combined
        ),
        "final_kinetic": (
            final_kinetic
        ),
    }


# =====================================================================
# Experiment driver
# =====================================================================

def run_experiment():
    """
    Run all grid / spatial-mode / epsilon combinations.
    """

    results = []

    total_cases = (
        len(NX_VALUES)
        * len(SPATIAL_MODES)
        * len(EPSILON_TARGETS)
    )

    case_number = 0

    print()
    print(
        "Experiment 005c: Grid convergence"
    )
    print(
        "----------------------------------"
    )
    print(
        f"Total cases: {total_cases}"
    )
    print()

    for nx in NX_VALUES:

        for spatial_mode in SPATIAL_MODES:

            for epsilon in EPSILON_TARGETS:

                case_number += 1

                print(
                    f"[{case_number:02d}/{total_cases:02d}] "
                    f"NX={nx:4d} "
                    f"k={spatial_mode:2d} "
                    f"epsilon={epsilon:.3f}"
                )

                result = run_case(
                    nx,
                    spatial_mode,
                    epsilon,
                )

                results.append(
                    result
                )

                print(
                    "    "
                    f"cells/wavelength="
                    f"{result['cells_per_wavelength']:.2f} "
                    f"final="
                    f"{result['final_combined']:.6f} "
                    f"max="
                    f"{result['max_combined']:.6f}"
                )

    return results


# =====================================================================
# Save CSV
# =====================================================================

def save_csv(
    results,
    output_path,
):
    """
    Save all numerical results as CSV.
    """

    if not results:
        return

    fieldnames = list(
        results[0].keys()
    )

    with open(
        output_path,
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for result in results:

            writer.writerow(
                result
            )


# =====================================================================
# Save text summary
# =====================================================================

def save_summary(
    results,
    output_path,
):
    """
    Save a human-readable summary.
    """

    with open(
        output_path,
        "w",
    ) as handle:

        handle.write(
            "Experiment 005c: Grid convergence\n"
        )

        handle.write(
            "====================================\n\n"
        )

        handle.write(
            "Definitions\n"
        )

        handle.write(
            "-----------\n"
        )

        handle.write(
            "L = DOMAIN_LENGTH / spatial_mode\n"
        )

        handle.write(
            "tau = epsilon * L / REFERENCE_SPEED\n"
        )

        handle.write(
            "epsilon = tau * REFERENCE_SPEED / L\n"
        )

        handle.write(
            "cells_per_wavelength = L / dx\n\n"
        )

        handle.write(
            "The spatial mode directly controls the "
            "wavelength of the initial perturbation.\n\n"
        )

        for result in results:

            handle.write(
                f"NX={result['nx']:4d} "
                f"k={result['spatial_mode']:2d} "
                f"epsilon={result['epsilon_target']:.3f} "
                f"L={result['L']:.8f} "
                f""
                f"cells/wavelength="
                f"{result['cells_per_wavelength']:.4f} "
                f""
                f"final="
                f"{result['final_combined']:.10f} "
                f""
                f"max="
                f"{result['max_combined']:.10f}\n"
            )


# =====================================================================
# Plot: grid convergence
# =====================================================================

def plot_grid_convergence(
    results,
    output_path,
):
    """
    Plot final hydrodynamic discrepancy versus NX.
    """

    plt.figure()

    for spatial_mode in SPATIAL_MODES:

        for epsilon in EPSILON_TARGETS:

            subset = [
                result
                for result in results
                if (
                    result["spatial_mode"]
                    == spatial_mode
                    and np.isclose(
                        result[
                            "epsilon_target"
                        ],
                        epsilon,
                    )
                )
            ]

            subset.sort(
                key=lambda item:
                item["nx"]
            )

            x_values = [
                result["nx"]
                for result in subset
            ]

            y_values = [
                result["final_combined"]
                for result in subset
            ]

            plt.plot(
                x_values,
                y_values,
                marker="o",
                label=(
                    f"k={spatial_mode}, "
                    f"eps={epsilon}"
                ),
            )

    plt.xlabel(
        "Spatial grid points NX"
    )

    plt.ylabel(
        "Final hydrodynamic discrepancy"
    )

    plt.title(
        "005c: Grid convergence"
    )

    plt.legend(
        fontsize=8
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()


# =====================================================================
# Plot: cells per wavelength
# =====================================================================

def plot_cells_per_wavelength(
    results,
    output_path,
):
    """
    Plot final discrepancy against the number of cells resolving
    the physical perturbation wavelength.
    """

    plt.figure()

    for epsilon in EPSILON_TARGETS:

        subset = [
            result
            for result in results
            if np.isclose(
                result[
                    "epsilon_target"
                ],
                epsilon,
            )
        ]

        subset.sort(
            key=lambda item: (
                item[
                    "spatial_mode"
                ],
                item["nx"],
            )
        )

        x_values = [
            result[
                "cells_per_wavelength"
            ]
            for result in subset
        ]

        y_values = [
            result[
                "final_combined"
            ]
            for result in subset
        ]

        plt.scatter(
            x_values,
            y_values,
            label=(
                f"epsilon={epsilon}"
            ),
        )

    plt.xlabel(
        "Cells per wavelength"
    )

    plt.ylabel(
        "Final hydrodynamic discrepancy"
    )

    plt.title(
        "005c: Resolution dependence"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()


# =====================================================================
# Plot: physical scale dependence
# =====================================================================

def plot_scale_dependence(
    results,
    output_path,
):
    """
    Plot final discrepancy versus physical wavelength.
    """

    plt.figure()

    for epsilon in EPSILON_TARGETS:

        subset = [
            result
            for result in results
            if np.isclose(
                result[
                    "epsilon_target"
                ],
                epsilon,
            )
        ]

        subset.sort(
            key=lambda item:
            item["L"]
        )

        wavelengths = [
            result["L"]
            for result in subset
        ]

        discrepancies = [
            result[
                "final_combined"
            ]
            for result in subset
        ]

        plt.plot(
            wavelengths,
            discrepancies,
            marker="o",
            label=(
                f"epsilon={epsilon}"
            ),
        )

    plt.xlabel(
        "Physical wavelength L"
    )

    plt.ylabel(
        "Final hydrodynamic discrepancy"
    )

    plt.title(
        "005c: Scale dependence"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()


# =====================================================================
# Save outputs
# =====================================================================

def save_outputs(
    results,
):
    """
    Save CSV, text summary and plots.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_csv(
        results,
        OUTPUT_DIR
        / "results.csv",
    )

    save_summary(
        results,
        OUTPUT_DIR
        / "summary.txt",
    )

    plot_grid_convergence(
        results,
        OUTPUT_DIR
        / "grid_convergence.png",
    )

    plot_cells_per_wavelength(
        results,
        OUTPUT_DIR
        / "cells_per_wavelength.png",
    )

    plot_scale_dependence(
        results,
        OUTPUT_DIR
        / "scale_dependence.png",
    )


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":

    results = run_experiment()

    save_outputs(
        results
    )

    print()
    print(
        "Experiment complete."
    )

    print(
        f"Results written to: "
        f"{OUTPUT_DIR}"
    )
