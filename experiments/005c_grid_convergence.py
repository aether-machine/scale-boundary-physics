"""
Experiment 005c: Grid convergence at controlled scale separation.

Purpose
-------
Test whether the kinetic-to-hydrodynamic discrepancy observed in
Experiment 005b survives spatial grid refinement.

The experiment varies:

    NX              spatial resolution
    spatial_mode    perturbation wavenumber k
    epsilon         Knudsen-like scale ratio

with

    L = DOMAIN_LENGTH / spatial_mode
    tau = epsilon * L / REFERENCE_SPEED

The important methodological point is that spatial_mode controls the
actual wavelength of the initial perturbation:

    wavelength = DOMAIN_LENGTH / spatial_mode

so the scale parameter L corresponds to a physical feature of the
initial condition rather than merely changing the relaxation time.

The experiment compares two microscopic states that have identical
initial hydrodynamic moments but differ in higher-order velocity-space
structure.

If the resulting hydrodynamic discrepancy persists under spatial grid
refinement, that provides stronger evidence that the effect is not
simply a spatial discretization artifact.

This remains a toy BGK kinetic model, not a derivation of Navier-Stokes.
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

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

OUTPUT_DIR = Path("results/005c_grid_convergence")


# ---------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------


def velocity_grid():
    """
    Construct a uniform velocity grid.

    The endpoint is excluded so that the grid spacing is uniform.
    """

    dv = (V_MAX - V_MIN) / NV

    v = V_MIN + dv * np.arange(NV)

    return v, dv


def spatial_grid(nx):
    """
    Construct a periodic spatial grid.
    """

    dx = DOMAIN_LENGTH / nx

    x = dx * np.arange(nx)

    return x, dx


# ---------------------------------------------------------------------
# Maxwellian
# ---------------------------------------------------------------------


def maxwellian(
    density,
    velocity,
    temperature,
    v,
):
    """
    Construct a 1-D Maxwellian in velocity space.

    Parameters
    ----------
    density:
        Spatial density array with shape (NX,).

    velocity:
        Spatial mean velocity array with shape (NX,).

    temperature:
        Spatial temperature array with shape (NX,).

    v:
        1-D velocity grid with shape (NV,).

    Returns
    -------
    f:
        Distribution with shape (NX, NV).
    """

    density = np.asarray(density)
    velocity = np.asarray(velocity)
    temperature = np.asarray(temperature)
    v = np.asarray(v)

    if density.ndim != 1:
        raise ValueError(
            "density must be 1-D with shape (NX,), "
            f"got {density.shape}"
        )

    if velocity.ndim != 1:
        raise ValueError(
            "velocity must be 1-D with shape (NX,), "
            f"got {velocity.shape}"
        )

    if temperature.ndim != 1:
        raise ValueError(
            "temperature must be 1-D with shape (NX,), "
            f"got {temperature.shape}"
        )

    if v.ndim != 1:
        raise ValueError(
            "v must be 1-D with shape (NV,), "
            f"got {v.shape}"
        )

    if not (
        density.shape
        == velocity.shape
        == temperature.shape
    ):
        raise ValueError(
            "density, velocity and temperature "
            "must have identical shapes"
        )

    temperature = np.maximum(
        temperature,
        1.0e-8,
    )

    v_grid = v[None, :]

    prefactor = (
        density[:, None]
        / np.sqrt(
            2.0
            * np.pi
            * temperature[:, None]
        )
    )

    exponent = -(
        v_grid
        - velocity[:, None]
    ) ** 2 / (
        2.0
        * temperature[:, None]
    )

    result = (
        prefactor
        * np.exp(exponent)
    )

    return result


# ---------------------------------------------------------------------
# Hydrodynamic moments
# ---------------------------------------------------------------------


def hydrodynamic_moments(
    f,
    v,
    dv,
):
    """
    Calculate density, velocity and temperature.

    The distribution is assumed to be normalized such that

        rho = integral f dv

        rho*u = integral v*f dv

        rho*(u^2 + T) = integral v^2*f dv
    """

    f = np.asarray(f)
    v = np.asarray(v)

    if f.ndim != 2:
        raise ValueError(
            "f must be 2-D with shape (NX, NV), "
            f"got {f.shape}"
        )

    if v.ndim != 1:
        raise ValueError(
            "v must be 1-D with shape (NV,), "
            f"got {v.shape}"
        )

    if f.shape[1] != v.size:
        raise ValueError(
            "Distribution/velocity-grid mismatch: "
            f"f has shape {f.shape}, "
            f"v has shape {v.shape}"
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
        - velocity**2
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
    Return RMS differences between hydrodynamic moments.
    """

    rho_a, u_a, T_a = hydrodynamic_moments(
        f_a,
        v,
        dv,
    )

    rho_b, u_b, T_b = hydrodynamic_moments(
        f_b,
        v,
        dv,
    )

    density_difference = np.sqrt(
        np.mean(
            (rho_a - rho_b) ** 2
        )
    )

    velocity_difference = np.sqrt(
        np.mean(
            (u_a - u_b) ** 2
        )
    )

    temperature_difference = np.sqrt(
        np.mean(
            (T_a - T_b) ** 2
        )
    )

    combined_difference = np.sqrt(
        density_difference**2
        + velocity_difference**2
        + temperature_difference**2
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
    L2 difference in the full kinetic distribution.
    """

    if f_a.shape != f_b.shape:
        raise ValueError(
            "Kinetic distributions must have identical shapes: "
            f"{f_a.shape} != {f_b.shape}"
        )

    return np.sqrt(
        np.sum(
            (f_a - f_b) ** 2
            * dv
        )
    )


# ---------------------------------------------------------------------
# Moment matching
# ---------------------------------------------------------------------


def match_hydrodynamic_moments(
    f_reference,
    f_target,
    v,
    dv,
):
    """
    Correct f_target so that its density, momentum and second velocity
    moment match f_reference.

    The velocity coordinate is scaled to approximately [-1, 1] before
    constructing the moment system. This improves numerical conditioning
    without changing the resulting moment constraints.
    """

    f_reference = np.asarray(
        f_reference
    )

    f_target = np.asarray(
        f_target
    )

    v = np.asarray(v)

    if f_reference.ndim != 2:
        raise ValueError(
            "f_reference must be 2-D, "
            f"got {f_reference.shape}"
        )

    if f_target.ndim != 2:
        raise ValueError(
            "f_target must be 2-D, "
            f"got {f_target.shape}"
        )

    if f_reference.shape != f_target.shape:
        raise ValueError(
            "f_reference and f_target must have "
            "identical shapes: "
            f"{f_reference.shape} != {f_target.shape}"
        )

    if v.ndim != 1:
        raise ValueError(
            "v must be 1-D, "
            f"got {v.shape}"
        )

    if f_target.shape[1] != v.size:
        raise ValueError(
            "Distribution/velocity-grid mismatch: "
            f"f_target has shape {f_target.shape}, "
            f"v has shape {v.shape}"
        )

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

    # 3 x 3 moment matrix.
    moment_matrix = (
        basis * dv
    ) @ basis.T

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
                    * s**2
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
            + coefficients[2] * s**2
        )

        corrected[i] = (
            f_target[i]
            + correction
        )

    return corrected


# ---------------------------------------------------------------------
# Initial distributions
# ---------------------------------------------------------------------


def initial_distributions(
    nx,
    spatial_mode,
):
    """
    Construct two kinetic states A and B.

    State A is a local Maxwellian.

    State B contains a higher-order velocity-space perturbation which is
    then corrected so that density, momentum and second velocity moments
    initially match state A.

    The spatial wavelength of the perturbation is controlled explicitly
    by spatial_mode:

        wavelength = DOMAIN_LENGTH / spatial_mode

    IMPORTANT
    ---------
    Spatial macroscopic fields are kept 1-D with shape (NX,).

    The kinetic distribution is 2-D with shape (NX, NV).

    No spatial/velocity meshgrid is used for the macroscopic fields.
    """

    x, dx = spatial_grid(nx)

    v, dv = velocity_grid()

    # --------------------------------------------------------------
    # Macroscopic background state
    # --------------------------------------------------------------
    #
    # x has shape (NX,).
    #
    # Therefore density, velocity and temperature also have shape
    # (NX,).
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

    expected_spatial_shape = (
        nx,
    )

    if density.shape != expected_spatial_shape:
        raise RuntimeError(
            "density has unexpected shape: "
            f"{density.shape}; "
            f"expected {expected_spatial_shape}"
        )

    if velocity.shape != expected_spatial_shape:
        raise RuntimeError(
            "velocity has unexpected shape: "
            f"{velocity.shape}; "
            f"expected {expected_spatial_shape}"
        )

    if temperature.shape != expected_spatial_shape:
        raise RuntimeError(
            "temperature has unexpected shape: "
            f"{temperature.shape}; "
            f"expected {expected_spatial_shape}"
        )

    if v.shape != (NV,):
        raise RuntimeError(
            "velocity grid has unexpected shape: "
            f"{v.shape}; "
            f"expected {(NV,)}"
        )

    # --------------------------------------------------------------
    # Reference kinetic state
    # --------------------------------------------------------------

    f_a = maxwellian(
        density,
        velocity,
        temperature,
        v,
    )

    expected_distribution_shape = (
        nx,
        NV,
    )

    if f_a.shape != expected_distribution_shape:
        raise RuntimeError(
            "f_a has unexpected shape: "
            f"{f_a.shape}; "
            f"expected {expected_distribution_shape}"
        )

    # --------------------------------------------------------------
    # Hidden higher-order kinetic structure
    # --------------------------------------------------------------

    velocity_scale = max(
        float(np.max(np.abs(v))),
        1.0,
    )

    c = (
        v
        / velocity_scale
    )

    # Hermite-like fourth-order structure.
    velocity_shape = (
        c**4
        - 6.0 * c**2
        + 3.0
    )

    # Spatial structure has shape (NX,).
    spatial_shape = np.cos(
        spatial_mode * x
    )

    # Convert explicitly to:
    #
    # spatial_shape[:, None]  -> (NX, 1)
    # velocity_shape[None, :] -> (1, NV)
    #
    # Their product gives:
    #
    # (NX, NV)
    hidden = (
        HIDDEN_AMPLITUDE
        * spatial_shape[:, None]
        * velocity_shape[None, :]
        * f_a
    )

    if hidden.shape != expected_distribution_shape:
        raise RuntimeError(
            "hidden has unexpected shape: "
            f"{hidden.shape}; "
            f"expected {expected_distribution_shape}"
        )

    f_b = (
        f_a
        + hidden
    )

    # --------------------------------------------------------------
    # Match hydrodynamic moments
    # --------------------------------------------------------------

    f_b = match_hydrodynamic_moments(
        f_a,
        f_b,
        v,
        dv,
    )

    if f_b.shape != expected_distribution_shape:
        raise RuntimeError(
            "f_b has unexpected shape: "
            f"{f_b.shape}; "
            f"expected {expected_distribution_shape}"
        )

    # --------------------------------------------------------------
    # Verify initial hydrodynamic equivalence
    # --------------------------------------------------------------

    initial_difference = hydrodynamic_difference(
        f_a,
        f_b,
        v,
        dv,
    )

    if initial_difference[3] > 1.0e-10:
        raise RuntimeError(
            "Initial hydrodynamic moment matching failed: "
            f"combined difference = "
            f"{initial_difference[3]:.6e}"
        )

    return (
        x,
        dx,
        v,
        dv,
        f_a,
        f_b,
    )


# ---------------------------------------------------------------------
# Equilibrium
# ---------------------------------------------------------------------


def equilibrium(
    f,
    v,
    dv,
):
    """
    Construct the local Maxwellian equilibrium corresponding to f.
    """

    density, velocity, temperature = (
        hydrodynamic_moments(
            f,
            v,
            dv,
        )
    )

    return maxwellian(
        density,
        velocity,
        temperature,
        v,
    )


# ---------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------


def stream_upwind(
    f,
    v,
    dx,
    dt,
):
    """
    First-order upwind streaming step.

    Periodic boundary conditions are used in x.
    """

    f = np.asarray(f)
    v = np.asarray(v)

    if f.ndim != 2:
        raise ValueError(
            "stream_upwind expected a 2-D distribution "
            "with shape (NX, NV), "
            f"got {f.shape}"
        )

    if v.ndim != 1:
        raise ValueError(
            "stream_upwind expected a 1-D velocity grid, "
            f"got {v.shape}"
        )

    if f.shape[1] != v.size:
        raise ValueError(
            "Distribution/velocity-grid mismatch: "
            f"f has shape {f.shape}, "
            f"v has shape {v.shape}"
        )

    streamed = np.empty_like(
        f
    )

    for j, velocity in enumerate(v):

        column = f[:, j]

        if velocity > 0.0:

            streamed[:, j] = (
                column
                - velocity
                * dt
                / dx
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
                - velocity
                * dt
                / dx
                * (
                    np.roll(
                        column,
                        -1,
                    )
                    - column
                )
            )

        else:

            streamed[:, j] = column

    return streamed


# ---------------------------------------------------------------------
# BGK evolution
# ---------------------------------------------------------------------


def bgk_step(
    f,
    v,
    dv,
    dx,
    dt,
    tau,
):
    """
    Advance one BGK kinetic timestep.

    The implementation uses operator splitting:

        streaming
        +
        BGK relaxation
    """

    if tau <= 0.0:
        raise ValueError(
            f"tau must be positive, got {tau}"
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

    # Guard against tiny negative numerical values.
    f_new = np.maximum(
        f_new,
        0.0,
    )

    return f_new


# ---------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------


def run_case(
    nx,
    spatial_mode,
    epsilon,
):
    """
    Run one grid / wavelength / epsilon combination.

    Returns a dictionary containing the final and maximum
    hydrodynamic discrepancies.
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

    initial_kinetic = kinetic_difference(
        f_a,
        f_b,
        dv,
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

        # Defensive shape check.
        expected_shape = (
            nx,
            NV,
        )

        if f_a.shape != expected_shape:
            raise RuntimeError(
                "f_a changed to unexpected shape "
                f"{f_a.shape}; "
                f"expected {expected_shape}"
            )

        if f_b.shape != expected_shape:
            raise RuntimeError(
                "f_b changed to unexpected shape "
                f"{f_b.shape}; "
                f"expected {expected_shape}"
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

    final_kinetic = kinetic_difference(
        f_a,
        f_b,
        dv,
    )

    epsilon_actual = (
        tau
        * REFERENCE_SPEED
        / L
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
            wavelength / dx
        ),
        "initial_kinetic": initial_kinetic,
        "initial_hydro": initial_combined,
        "max_density": max_density,
        "max_velocity": max_velocity,
        "max_temperature": max_temperature,
        "max_combined": max_combined,
        "final_density": final_density,
        "final_velocity": final_velocity,
        "final_temperature": final_temperature,
        "final_combined": final_combined,
        "final_kinetic": final_kinetic,
    }


# ---------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------


def run_experiment():
    """
    Run the complete grid-convergence experiment.
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


# ---------------------------------------------------------------------
# Save tabular results
# ---------------------------------------------------------------------


def save_csv(
    results,
    output_path,
):
    """
    Save numerical results to CSV.
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


def save_summary(
    results,
    output_path,
):
    """
    Save a human-readable text summary.
    """

    with open(
        output_path,
        "w",
    ) as handle:

        handle.write(
            "Experiment 005c: "
            "Grid convergence\n"
        )

        handle.write(
            "====================================\n\n"
        )

        handle.write(
            "Purpose\n"
        )

        handle.write(
            "-------\n"
        )

        handle.write(
            "Test whether kinetic-to-hydrodynamic "
            "discrepancy persists under spatial "
            "grid refinement.\n\n"
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
            "epsilon = tau * REFERENCE_SPEED / L\n\n"
        )

        handle.write(
            "The spatial perturbation itself uses "
            "spatial_mode, so its wavelength is L.\n\n"
        )

        for result in results:

            handle.write(
                f"NX={result['nx']:4d} "
                f"k={result['spatial_mode']:2d} "
                f"epsilon={result['epsilon_target']:.3f} "
                f"L={result['L']:.6f} "
                f"cells/wavelength="
                f"{result['cells_per_wavelength']:.2f} "
                f"final="
                f"{result['final_combined']:.8f} "
                f"max="
                f"{result['max_combined']:.8f}\n"
            )


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def plot_grid_convergence(
    results,
    output_path,
):
    """
    Plot final hydrodynamic discrepancy against spatial resolution.
    """

    plt.figure()

    for spatial_mode in SPATIAL_MODES:

        for epsilon in EPSILON_TARGETS:

            subset = [
                result
                for result in results
                if result["spatial_mode"]
                == spatial_mode
                and np.isclose(
                    result["epsilon_target"],
                    epsilon,
                )
            ]

            subset.sort(
                key=lambda item: item["nx"]
            )

            nx_values = [
                result["nx"]
                for result in subset
            ]

            discrepancy = [
                result["final_combined"]
                for result in subset
            ]

            plt.plot(
                nx_values,
                discrepancy,
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


def plot_cells_per_wavelength(
    results,
    output_path,
):
    """
    Plot final discrepancy against cells per wavelength.

    This is often more informative than plotting against NX because
    the relevant numerical resolution is the number of grid cells
    resolving the physical perturbation.
    """

    plt.figure()

    for epsilon in EPSILON_TARGETS:

        subset = [
            result
            for result in results
            if np.isclose(
                result["epsilon_target"],
                epsilon,
            )
        ]

        subset.sort(
            key=lambda item: (
                item["spatial_mode"],
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
            label=f"epsilon={epsilon}",
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
                result["epsilon_target"],
                epsilon,
            )
        ]

        subset.sort(
            key=lambda item: item["L"]
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
            label=f"epsilon={epsilon}",
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


# ---------------------------------------------------------------------
# Output handling
# ---------------------------------------------------------------------


def save_outputs(
    results,
):
    """
    Save CSV, summary and plots.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_csv(
        results,
        OUTPUT_DIR / "results.csv",
    )

    save_summary(
        results,
        OUTPUT_DIR / "summary.txt",
    )

    plot_grid_convergence(
        results,
        OUTPUT_DIR / "grid_convergence.png",
    )

    plot_cells_per_wavelength(
        results,
        OUTPUT_DIR / "cells_per_wavelength.png",
    )

    plot_scale_dependence(
        results,
        OUTPUT_DIR / "scale_dependence.png",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


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
