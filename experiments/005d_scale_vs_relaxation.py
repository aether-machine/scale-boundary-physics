"""
Experiment 005d: Scale versus relaxation time.

Purpose
-------
Separate the effects of physical spatial scale and kinetic relaxation
time in the kinetic-to-hydrodynamic discrepancy.

Experiment 005c varied

    L = DOMAIN_LENGTH / spatial_mode

while simultaneously imposing

    tau = epsilon * L / REFERENCE_SPEED.

That means physical scale and relaxation time changed together.

Experiment 005d breaks that coupling.

We independently vary:

    spatial_mode -> physical wavelength L
    tau          -> kinetic relaxation time

and measure the resulting hydrodynamic discrepancy.

The central quantity is therefore

    Delta = Delta(L, tau)

rather than Delta(L) along a prescribed tau/L trajectory.

This allows us to investigate four possibilities:

    A. discrepancy is primarily controlled by tau
    B. discrepancy is primarily controlled by L
    C. discrepancy is primarily controlled by epsilon = tau/L
    D. discrepancy depends on an interaction between L and tau

This remains a toy BGK kinetic model, not a derivation of
Navier-Stokes.

Scientific question
-------------------
Does the kinetic-to-hydrodynamic closure error exhibit a characteristic
scale boundary when physical wavelength and relaxation time are varied
independently?

The experiment is deliberately agnostic about the answer.
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

# A single well-resolved spatial grid is used initially.
#
# Grid convergence was tested separately in Experiment 005c.
#
# NX=512 gives:
#
#   k=1   -> 512 cells / wavelength
#   k=4   -> 128
#   k=8   -> 64
#   k=16  -> 32
#
# The k=24 case from 005c is omitted here initially because the purpose
# of 005d is parameter separation rather than pushing the shortest
# wavelength.

NX = 512

NV = 96

DOMAIN_LENGTH = 2.0 * np.pi

V_MIN = -4.0
V_MAX = 4.0

DT = 0.001
T_FINAL = 4.0

REFERENCE_SPEED = 1.0

# Physical scales.
#
# wavelength = DOMAIN_LENGTH / spatial_mode
#
SPATIAL_MODES = [
    1,
    4,
    8,
    16,
]

# Relaxation times are now specified directly.
#
# They are deliberately NOT calculated from L.
TAU_VALUES = [
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.20,
]

HIDDEN_AMPLITUDE = 0.35

OUTPUT_DIR = Path(
    "results/005d_scale_vs_relaxation"
)


# ---------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------


def velocity_grid():
    """
    Construct a uniform velocity grid.

    The endpoint is excluded so that the grid spacing is uniform.
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
    Construct a periodic spatial grid.
    """

    dx = (
        DOMAIN_LENGTH
        / nx
    )

    x = (
        dx
        * np.arange(nx)
    )

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
    Construct a local Maxwellian.

    density, velocity and temperature have shape (NX,).

    v has shape (NV,).

    The returned distribution has shape (NX, NV).
    """

    density = np.asarray(
        density
    )

    velocity = np.asarray(
        velocity
    )

    temperature = np.asarray(
        temperature
    )

    v = np.asarray(v)

    if density.ndim != 1:
        raise ValueError(
            "density must be 1-D, "
            f"got {density.shape}"
        )

    if velocity.ndim != 1:
        raise ValueError(
            "velocity must be 1-D, "
            f"got {velocity.shape}"
        )

    if temperature.ndim != 1:
        raise ValueError(
            "temperature must be 1-D, "
            f"got {temperature.shape}"
        )

    if v.ndim != 1:
        raise ValueError(
            "v must be 1-D, "
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

    return (
        prefactor
        * np.exp(exponent)
    )


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
    """

    f = np.asarray(f)
    v = np.asarray(v)

    if f.ndim != 2:
        raise ValueError(
            "f must be 2-D, "
            f"got {f.shape}"
        )

    if v.ndim != 1:
        raise ValueError(
            "v must be 1-D, "
            f"got {v.shape}"
        )

    if f.shape[1] != v.size:
        raise ValueError(
            "Distribution/velocity-grid mismatch: "
            f"f={f.shape}, v={v.shape}"
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

    rho_a, u_a, T_a = (
        hydrodynamic_moments(
            f_a,
            v,
            dv,
        )
    )

    rho_b, u_b, T_b = (
        hydrodynamic_moments(
            f_b,
            v,
            dv,
        )
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
    L2 difference between the complete kinetic states.
    """

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
            "f_reference must be 2-D"
        )

    if f_target.ndim != 2:
        raise ValueError(
            "f_target must be 2-D"
        )

    if f_reference.shape != f_target.shape:
        raise ValueError(
            "f_reference and f_target "
            "must have identical shapes"
        )

    if v.ndim != 1:
        raise ValueError(
            "v must be 1-D"
        )

    if f_target.shape[1] != v.size:
        raise ValueError(
            "Distribution/velocity-grid mismatch"
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
            s**2,
        ]
    )

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
    Construct kinetic states A and B.

    Both states have identical initial hydrodynamic moments but differ
    in higher-order velocity-space structure.
    """

    x, dx = spatial_grid(
        nx
    )

    v, dv = velocity_grid()

    # --------------------------------------------------------------
    # Macroscopic state
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
            f"density shape {density.shape}; "
            f"expected {expected_spatial_shape}"
        )

    if velocity.shape != expected_spatial_shape:
        raise RuntimeError(
            f"velocity shape {velocity.shape}; "
            f"expected {expected_spatial_shape}"
        )

    if temperature.shape != expected_spatial_shape:
        raise RuntimeError(
            f"temperature shape {temperature.shape}; "
            f"expected {expected_spatial_shape}"
        )

    # --------------------------------------------------------------
    # Reference Maxwellian
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
            f"f_a shape {f_a.shape}; "
            f"expected {expected_distribution_shape}"
        )

    # --------------------------------------------------------------
    # Hidden velocity-space structure
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
        c**4
        - 6.0 * c**2
        + 3.0
    )

    spatial_shape = np.cos(
        spatial_mode * x
    )

    hidden = (
        HIDDEN_AMPLITUDE
        * spatial_shape[:, None]
        * velocity_shape[None, :]
        * f_a
    )

    if hidden.shape != expected_distribution_shape:
        raise RuntimeError(
            f"hidden shape {hidden.shape}; "
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
            f"f_b shape {f_b.shape}; "
            f"expected {expected_distribution_shape}"
        )

    # Verify initial equality.
    initial_difference = (
        hydrodynamic_difference(
            f_a,
            f_b,
            v,
            dv,
        )
    )

    if initial_difference[3] > 1.0e-10:
        raise RuntimeError(
            "Initial hydrodynamic matching failed: "
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
    Construct the local Maxwellian corresponding to f.
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
    First-order upwind streaming step with periodic boundaries.
    """

    f = np.asarray(f)
    v = np.asarray(v)

    if f.ndim != 2:
        raise ValueError(
            "f must be 2-D, "
            f"got {f.shape}"
        )

    if v.ndim != 1:
        raise ValueError(
            "v must be 1-D, "
            f"got {v.shape}"
        )

    if f.shape[1] != v.size:
        raise ValueError(
            "Distribution/velocity mismatch: "
            f"f={f.shape}, v={v.shape}"
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

    # Guard against tiny negative values.
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
    tau,
):
    """
    Run one physical-scale / relaxation-time combination.

    Unlike Experiment 005c, tau is supplied independently of L.
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

    n_steps = int(
        round(
            T_FINAL / DT
        )
    )

    # Dimensionless scale-separation ratio.
    epsilon = (
        tau
        * REFERENCE_SPEED
        / L
    )

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

        expected_shape = (
            nx,
            NV,
        )

        if f_a.shape != expected_shape:
            raise RuntimeError(
                f"f_a changed shape to "
                f"{f_a.shape}; "
                f"expected {expected_shape}"
            )

        if f_b.shape != expected_shape:
            raise RuntimeError(
                f"f_b changed shape to "
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

    final_kinetic = (
        kinetic_difference(
            f_a,
            f_b,
            dv,
        )
    )

    return {
        "nx": nx,
        "spatial_mode": spatial_mode,
        "wavelength": wavelength,
        "L": L,
        "tau": tau,
        "epsilon": epsilon,
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
    Run the complete two-dimensional L/tau parameter sweep.
    """

    results = []

    total_cases = (
        len(SPATIAL_MODES)
        * len(TAU_VALUES)
    )

    case_number = 0

    print()
    print(
        "Experiment 005d: Scale vs relaxation"
    )
    print(
        "-------------------------------------"
    )
    print(
        f"NX={NX}"
    )
    print(
        f"Total cases: {total_cases}"
    )
    print()

    for spatial_mode in SPATIAL_MODES:

        for tau in TAU_VALUES:

            case_number += 1

            wavelength = (
                DOMAIN_LENGTH
                / spatial_mode
            )

            epsilon = (
                tau
                * REFERENCE_SPEED
                / wavelength
            )

            print(
                f"[{case_number:02d}/{total_cases:02d}] "
                f"k={spatial_mode:2d} "
                f"L={wavelength:.6f} "
                f"tau={tau:.5f} "
                f"epsilon={epsilon:.5f}"
            )

            result = run_case(
                NX,
                spatial_mode,
                tau,
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
# Save CSV
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

        writer.writerows(
            results
        )


# ---------------------------------------------------------------------
# Save summary
# ---------------------------------------------------------------------


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
            "Experiment 005d: "
            "Scale vs relaxation\n"
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
            "epsilon = tau * REFERENCE_SPEED / L\n"
        )

        handle.write(
            "cells_per_wavelength = L / dx\n\n"
        )

        handle.write(
            "Unlike Experiment 005c, tau is "
            "varied independently of L.\n\n"
        )

        for result in results:

            handle.write(
                f"NX={result['nx']:4d} "
                f"k={result['spatial_mode']:2d} "
                f"L={result['L']:.8f} "
                f"tau={result['tau']:.5f} "
                f"epsilon={result['epsilon']:.5f} "
                f"cells/wavelength="
                f"{result['cells_per_wavelength']:.4f} "
                f"final="
                f"{result['final_combined']:.10f} "
                f"max="
                f"{result['max_combined']:.10f}\n"
            )


# ---------------------------------------------------------------------
# Plot 1: discrepancy vs tau
# ---------------------------------------------------------------------


def plot_discrepancy_vs_tau(
    results,
    output_path,
):
    """
    Plot final discrepancy against tau.

    Each line represents one physical wavelength.
    """

    plt.figure()

    for spatial_mode in SPATIAL_MODES:

        subset = [
            result
            for result in results
            if result["spatial_mode"]
            == spatial_mode
        ]

        subset.sort(
            key=lambda item: item["tau"]
        )

        tau_values = [
            result["tau"]
            for result in subset
        ]

        discrepancy = [
            result["final_combined"]
            for result in subset
        ]

        plt.plot(
            tau_values,
            discrepancy,
            marker="o",
            label=(
                f"k={spatial_mode}"
            ),
        )

    plt.xlabel(
        "Relaxation time tau"
    )

    plt.ylabel(
        "Final hydrodynamic discrepancy"
    )

    plt.title(
        "005d: Discrepancy vs relaxation time"
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
# Plot 2: discrepancy vs L
# ---------------------------------------------------------------------


def plot_discrepancy_vs_scale(
    results,
    output_path,
):
    """
    Plot final discrepancy against physical wavelength.

    Each line represents one relaxation time.
    """

    plt.figure()

    for tau in TAU_VALUES:

        subset = [
            result
            for result in results
            if np.isclose(
                result["tau"],
                tau,
            )
        ]

        subset.sort(
            key=lambda item: item["L"]
        )

        L_values = [
            result["L"]
            for result in subset
        ]

        discrepancy = [
            result["final_combined"]
            for result in subset
        ]

        plt.plot(
            L_values,
            discrepancy,
            marker="o",
            label=(
                f"tau={tau:g}"
            ),
        )

    plt.xlabel(
        "Physical wavelength L"
    )

    plt.ylabel(
        "Final hydrodynamic discrepancy"
    )

    plt.title(
        "005d: Discrepancy vs physical scale"
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


# ---------------------------------------------------------------------
# Plot 3: parameter-space map
# ---------------------------------------------------------------------


def plot_parameter_map(
    results,
    output_path,
):
    """
    Plot the final discrepancy across the two-dimensional
    (L, tau) parameter space.

    Each point is annotated by epsilon = tau/L.
    """

    plt.figure()

    L_values = np.array(
        [
            result["L"]
            for result in results
        ]
    )

    tau_values = np.array(
        [
            result["tau"]
            for result in results
        ]
    )

    discrepancy = np.array(
        [
            result["final_combined"]
            for result in results
        ]
    )

    scatter = plt.scatter(
        L_values,
        tau_values,
        c=discrepancy,
        s=90,
    )

    plt.colorbar(
        scatter,
        label="Final discrepancy",
    )

    for result in results:

        plt.annotate(
            f"{result['epsilon']:.3f}",
            (
                result["L"],
                result["tau"],
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )

    plt.xlabel(
        "Physical wavelength L"
    )

    plt.ylabel(
        "Relaxation time tau"
    )

    plt.title(
        "005d: Scale-relaxation parameter space"
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


# ---------------------------------------------------------------------
# Plot 4: discrepancy vs epsilon
# ---------------------------------------------------------------------


def plot_discrepancy_vs_epsilon(
    results,
    output_path,
):
    """
    Plot final discrepancy against epsilon=tau/L.

    This is useful for comparison with the conventional hydrodynamic
    scale-separation parameter.
    """

    plt.figure()

    for spatial_mode in SPATIAL_MODES:

        subset = [
            result
            for result in results
            if result["spatial_mode"]
            == spatial_mode
        ]

        subset.sort(
            key=lambda item: item["epsilon"]
        )

        epsilon_values = [
            result["epsilon"]
            for result in subset
        ]

        discrepancy = [
            result["final_combined"]
            for result in subset
        ]

        plt.plot(
            epsilon_values,
            discrepancy,
            marker="o",
            label=(
                f"k={spatial_mode}"
            ),
        )

    plt.xlabel(
        "epsilon = tau / L"
    )

    plt.ylabel(
        "Final hydrodynamic discrepancy"
    )

    plt.title(
        "005d: Discrepancy vs scale-separation ratio"
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
    Save all numerical and graphical results.
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

    plot_discrepancy_vs_tau(
        results,
        OUTPUT_DIR
        / "discrepancy_vs_tau.png",
    )

    plot_discrepancy_vs_scale(
        results,
        OUTPUT_DIR
        / "discrepancy_vs_scale.png",
    )

    plot_parameter_map(
        results,
        OUTPUT_DIR
        / "parameter_map.png",
    )

    plot_discrepancy_vs_epsilon(
        results,
        OUTPUT_DIR
        / "discrepancy_vs_epsilon.png",
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
