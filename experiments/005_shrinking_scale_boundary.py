"""
Experiment 005: Shrinking-Scale Closure Boundary

Purpose
-------
Investigate whether hydrodynamic closure deteriorates as the spatial
scale of the resolved structure becomes progressively smaller.

The kinetic model is the BGK-style model used in Experiments 004,
004b, and 004c:

    df/dt + v df/dx = (f_eq - f) / tau

Two kinetic states are constructed with identical initial
hydrodynamic moments:

    rho_A = rho_B
    u_A   = u_B
    T_A   = T_B

but different higher-order velocity-space structure.

The relaxation time is held fixed while the spatial wavelength of
the macroscopic structure is varied.

The central question is:

    Does hidden kinetic information become increasingly important
    as the macroscopic spatial scale approaches the kinetic scale?

This is a controlled model experiment. It is not yet a simulation
of the Navier-Stokes singularity and does not establish a physical
Knudsen-number threshold.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# Repository path
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

NX = 128
NV = 96

DOMAIN_LENGTH = 2.0 * np.pi

X_MIN = 0.0
X_MAX = DOMAIN_LENGTH

V_MIN = -4.0
V_MAX = 4.0

DT = 0.001
T_FINAL = 4.0

# Fixed kinetic relaxation time.
TAU = 0.05

# Spatial modes.
# Larger k -> smaller characteristic spatial scale.
SPATIAL_MODES = [
    1,
    2,
    4,
    8,
    12,
    16,
    24,
]

HIDDEN_AMPLITUDE = 0.35


# ---------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------

def make_grids():
    x = np.linspace(
        X_MIN,
        X_MAX,
        NX,
        endpoint=False,
    )

    v = np.linspace(
        V_MIN,
        V_MAX,
        NV,
    )

    dx = DOMAIN_LENGTH / NX
    dv = v[1] - v[0]

    return x, v, dx, dv


# ---------------------------------------------------------------------
# Maxwellian
# ---------------------------------------------------------------------

def maxwellian(
    v,
    density,
    mean_velocity,
    temperature,
):
    temperature = np.maximum(
        temperature,
        1e-10,
    )

    return (
        density[:, None]
        / np.sqrt(
            2.0
            * np.pi
            * temperature[:, None]
        )
        * np.exp(
            -(
                v[None, :]
                - mean_velocity[:, None]
            ) ** 2
            / (
                2.0
                * temperature[:, None]
            )
        )
    )


# ---------------------------------------------------------------------
# Initial hydrodynamic fields
# ---------------------------------------------------------------------

def initial_hydrodynamic_fields(
    x,
    spatial_mode,
):
    density = (
        1.0
        + 0.15
        * np.sin(
            spatial_mode * x
        )
    )

    mean_velocity = (
        0.30
        * np.cos(
            spatial_mode * x
        )
    )

    temperature = (
        1.0
        + 0.10
        * np.sin(
            spatial_mode * x
        )
    )

    return (
        density,
        mean_velocity,
        temperature,
    )


# ---------------------------------------------------------------------
# Hidden velocity-space structure
# ---------------------------------------------------------------------

def hidden_velocity_shape(v):
    return (
        (
            v ** 4
            - 6.0 * v ** 2
            + 3.0
        )
        * np.exp(
            -0.5 * v ** 2
        )
    )


# ---------------------------------------------------------------------
# Hydrodynamic moments
# ---------------------------------------------------------------------

def hydrodynamic_moments(
    f,
    v,
    dv,
):
    density = (
        np.sum(
            f,
            axis=1,
        )
        * dv
    )

    momentum = (
        np.sum(
            f
            * v[None, :],
            axis=1,
        )
        * dv
    )

    mean_velocity = (
        momentum
        / np.maximum(
            density,
            1e-14,
        )
    )

    energy = (
        np.sum(
            f
            * v[None, :] ** 2,
            axis=1,
        )
        * dv
    )

    temperature = (
        energy
        / np.maximum(
            density,
            1e-14,
        )
        - mean_velocity ** 2
    )

    return (
        density,
        mean_velocity,
        temperature,
    )


# ---------------------------------------------------------------------
# Moment matching
# ---------------------------------------------------------------------

def match_hydrodynamic_moments(
    reference,
    candidate,
    v,
    dv,
):
    """
    Correct candidate so density, momentum and energy match reference.
    """

    (
        ref_density,
        ref_velocity,
        ref_temperature,
    ) = hydrodynamic_moments(
        reference,
        v,
        dv,
    )

    target_momentum = (
        ref_density
        * ref_velocity
    )

    target_energy = (
        ref_density
        * (
            ref_temperature
            + ref_velocity ** 2
        )
    )

    (
        current_density,
        current_velocity,
        current_temperature,
    ) = hydrodynamic_moments(
        candidate,
        v,
        dv,
    )

    current_momentum = (
        current_density
        * current_velocity
    )

    current_energy = (
        current_density
        * (
            current_temperature
            + current_velocity ** 2
        )
    )

    moment_matrix = np.array(
        [
            [
                np.sum(
                    np.ones_like(v)
                ) * dv,
                np.sum(v) * dv,
                np.sum(v ** 2) * dv,
            ],
            [
                np.sum(v) * dv,
                np.sum(v ** 2) * dv,
                np.sum(v ** 3) * dv,
            ],
            [
                np.sum(v ** 2) * dv,
                np.sum(v ** 3) * dv,
                np.sum(v ** 4) * dv,
            ],
        ]
    )

    inverse = np.linalg.inv(
        moment_matrix
    )

    corrected = candidate.copy()

    for i in range(
        candidate.shape[0]
    ):

        target = np.array(
            [
                ref_density[i],
                target_momentum[i],
                target_energy[i],
            ]
        )

        current = np.array(
            [
                current_density[i],
                current_momentum[i],
                current_energy[i],
            ]
        )

        coefficients = (
            inverse
            @ (
                target
                - current
            )
        )

        correction = (
            coefficients[0]
            + coefficients[1] * v
            + coefficients[2] * v ** 2
        )

        corrected[i, :] += correction

    return corrected


# ---------------------------------------------------------------------
# Initial kinetic states
# ---------------------------------------------------------------------

def construct_initial_states(
    x,
    v,
    dv,
    spatial_mode,
):
    (
        density,
        velocity,
        temperature,
    ) = initial_hydrodynamic_fields(
        x,
        spatial_mode,
    )

    f_a = maxwellian(
        v,
        density,
        velocity,
        temperature,
    )

    shape = hidden_velocity_shape(v)

    perturbation = (
        HIDDEN_AMPLITUDE
        * density[:, None]
        * shape[None, :]
        * np.exp(
            -0.5
            * (
                v[None, :]
                - velocity[:, None]
            ) ** 2
            / temperature[:, None]
        )
    )

    f_b = f_a + perturbation

    f_b = np.maximum(
        f_b,
        1e-12,
    )

    f_b = match_hydrodynamic_moments(
        f_a,
        f_b,
        v,
        dv,
    )

    f_b = np.maximum(
        f_b,
        1e-12,
    )

    f_b = match_hydrodynamic_moments(
        f_a,
        f_b,
        v,
        dv,
    )

    return f_a, f_b


# ---------------------------------------------------------------------
# Equilibrium distribution
# ---------------------------------------------------------------------

def equilibrium_from_moments(
    f,
    v,
    dv,
):
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
        v,
        density,
        velocity,
        temperature,
    )


# ---------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------

def streaming_step(
    f,
    v,
    dx,
    dt,
):
    streamed = np.empty_like(f)

    for j, velocity in enumerate(v):

        if velocity >= 0.0:

            streamed[:, j] = (
                f[:, j]
                - velocity
                * dt
                / dx
                * (
                    f[:, j]
                    - np.roll(
                        f[:, j],
                        1,
                    )
                )
            )

        else:

            streamed[:, j] = (
                f[:, j]
                - velocity
                * dt
                / dx
                * (
                    np.roll(
                        f[:, j],
                        -1,
                    )
                    - f[:, j]
                )
            )

    return streamed


# ---------------------------------------------------------------------
# Kinetic step
# ---------------------------------------------------------------------

def kinetic_step(
    f,
    v,
    dx,
    dv,
    dt,
    tau,
):
    streamed = streaming_step(
        f,
        v,
        dx,
        dt,
    )

    equilibrium = (
        equilibrium_from_moments(
            streamed,
            v,
            dv,
        )
    )

    relaxation_factor = np.exp(
        -dt / tau
    )

    updated = (
        equilibrium
        + (
            streamed
            - equilibrium
        )
        * relaxation_factor
    )

    return np.maximum(
        updated,
        1e-12,
    )


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def hydrodynamic_difference(
    f_a,
    f_b,
    v,
    dv,
):
    (
        rho_a,
        u_a,
        t_a,
    ) = hydrodynamic_moments(
        f_a,
        v,
        dv,
    )

    (
        rho_b,
        u_b,
        t_b,
    ) = hydrodynamic_moments(
        f_b,
        v,
        dv,
    )

    rho_diff = np.sqrt(
        np.mean(
            (rho_a - rho_b) ** 2
        )
    )

    velocity_diff = np.sqrt(
        np.mean(
            (u_a - u_b) ** 2
        )
    )

    temperature_diff = np.sqrt(
        np.mean(
            (t_a - t_b) ** 2
        )
    )

    combined = np.sqrt(
        rho_diff ** 2
        + velocity_diff ** 2
        + temperature_diff ** 2
    )

    return (
        rho_diff,
        velocity_diff,
        temperature_diff,
        combined,
    )


def kinetic_difference(
    f_a,
    f_b,
    dx,
    dv,
):
    return np.sqrt(
        np.sum(
            (f_a - f_b) ** 2
        )
        * dx
        * dv
    )


# ---------------------------------------------------------------------
# Effective scale-separation parameter
# ---------------------------------------------------------------------

def characteristic_scale(
    spatial_mode,
):
    """
    L = 2*pi/k
    """
    return (
        DOMAIN_LENGTH
        / spatial_mode
    )


def effective_scale_parameter(
    spatial_mode,
):
    """
    A dimensionless proxy for scale separation.

    This is NOT a physical Knudsen number.

    We use:

        epsilon = tau / T_wave

    with

        T_wave = L / v_ref

    and v_ref = 1.

    Therefore:

        epsilon = tau / L

    Smaller epsilon means stronger separation between the kinetic
    relaxation time and the macroscopic spatial evolution scale.
    """

    L = characteristic_scale(
        spatial_mode
    )

    reference_speed = 1.0

    macroscopic_time = (
        L
        / reference_speed
    )

    return (
        TAU
        / macroscopic_time
    )


# ---------------------------------------------------------------------
# Single spatial-scale run
# ---------------------------------------------------------------------

def run_single_mode(
    spatial_mode,
    f_a_initial,
    f_b_initial,
    v,
    dx,
    dv,
):
    f_a = f_a_initial.copy()
    f_b = f_b_initial.copy()

    n_steps = int(
        round(
            T_FINAL / DT
        )
    )

    sample_interval = max(
        1,
        n_steps // 500,
    )

    times = []

    rho_history = []
    velocity_history = []
    temperature_history = []
    combined_history = []
    kinetic_history = []

    for step in range(
        n_steps + 1
    ):

        if (
            step % sample_interval == 0
            or step == n_steps
        ):

            (
                rho_diff,
                velocity_diff,
                temperature_diff,
                combined,
            ) = hydrodynamic_difference(
                f_a,
                f_b,
                v,
                dv,
            )

            times.append(
                step * DT
            )

            rho_history.append(
                rho_diff
            )

            velocity_history.append(
                velocity_diff
            )

            temperature_history.append(
                temperature_diff
            )

            combined_history.append(
                combined
            )

            kinetic_history.append(
                kinetic_difference(
                    f_a,
                    f_b,
                    dx,
                    dv,
                )
            )

        if step < n_steps:

            f_a = kinetic_step(
                f_a,
                v,
                dx,
                dv,
                DT,
                TAU,
            )

            f_b = kinetic_step(
                f_b,
                v,
                dx,
                dv,
                DT,
                TAU,
            )

    return {
        "spatial_mode": spatial_mode,
        "L": characteristic_scale(
            spatial_mode
        ),
        "epsilon": effective_scale_parameter(
            spatial_mode
        ),
        "times": np.asarray(times),
        "rho": np.asarray(
            rho_history
        ),
        "velocity": np.asarray(
            velocity_history
        ),
        "temperature": np.asarray(
            temperature_history
        ),
        "combined": np.asarray(
            combined_history
        ),
        "kinetic": np.asarray(
            kinetic_history
        ),
    }


# ---------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------

def run_experiment():
    x, v, dx, dv = make_grids()

    results = []

    for spatial_mode in SPATIAL_MODES:

        print()
        print(
            f"Running spatial mode k="
            f"{spatial_mode}..."
        )

        f_a, f_b = (
            construct_initial_states(
                x,
                v,
                dv,
                spatial_mode,
            )
        )

        initial_difference = (
            hydrodynamic_difference(
                f_a,
                f_b,
                v,
                dv,
            )
        )

        initial_kinetic = (
            kinetic_difference(
                f_a,
                f_b,
                dx,
                dv,
            )
        )

        print(
            f"Initial hydro difference = "
            f"{initial_difference[3]:.6e}"
        )

        print(
            f"Initial kinetic difference = "
            f"{initial_kinetic:.6e}"
        )

        result = run_single_mode(
            spatial_mode,
            f_a,
            f_b,
            v,
            dx,
            dv,
        )

        result["initial_hydro"] = (
            initial_difference[3]
        )

        result["initial_kinetic"] = (
            initial_kinetic
        )

        results.append(
            result
        )

    return results


# ---------------------------------------------------------------------
# Save outputs
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
        / "experiment_005_summary.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "spatial_mode,"
            "L,"
            "epsilon,"
            "initial_hydro,"
            "initial_kinetic,"
            "maximum_density,"
            "final_density,"
            "maximum_velocity,"
            "final_velocity,"
            "maximum_temperature,"
            "final_temperature,"
            "maximum_combined,"
            "final_combined,"
            "maximum_kinetic,"
            "final_kinetic\n"
        )

        for result in results:

            handle.write(
                f"{result['spatial_mode']},"
                f"{result['L']:.12e},"
                f"{result['epsilon']:.12e},"
                f"{result['initial_hydro']:.12e},"
                f"{result['initial_kinetic']:.12e},"
                f"{np.max(result['rho']):.12e},"
                f"{result['rho'][-1]:.12e},"
                f"{np.max(result['velocity']):.12e},"
                f"{result['velocity'][-1]:.12e},"
                f"{np.max(result['temperature']):.12e},"
                f"{result['temperature'][-1]:.12e},"
                f"{np.max(result['combined']):.12e},"
                f"{result['combined'][-1]:.12e},"
                f"{np.max(result['kinetic']):.12e},"
                f"{result['kinetic'][-1]:.12e}\n"
            )

    # ---------------------------------------------------------------
    # Text summary
    # ---------------------------------------------------------------

    summary_path = (
        results_dir
        / "experiment_005_summary.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "Experiment 005: "
            "Shrinking-Scale Closure Boundary\n"
        )

        handle.write(
            "============================================\n\n"
        )

        handle.write(
            f"Spatial grid: {NX}\n"
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
            f"Relaxation time tau: {TAU}\n\n"
        )

        handle.write(
            "Results\n"
        )

        handle.write(
            "-------\n"
        )

        handle.write(
            "spatial_mode,"
            "L,"
            "epsilon,"
            "initial_hydro,"
            "initial_kinetic,"
            "maximum_density,"
            "final_density,"
            "maximum_velocity,"
            "final_velocity,"
            "maximum_temperature,"
            "final_temperature,"
            "maximum_combined,"
            "final_combined,"
            "maximum_kinetic,"
            "final_kinetic\n"
        )

        for result in results:

            handle.write(
                f"{result['spatial_mode']},"
                f"{result['L']:.12e},"
                f"{result['epsilon']:.12e},"
                f"{result['initial_hydro']:.12e},"
                f"{result['initial_kinetic']:.12e},"
                f"{np.max(result['rho']):.12e},"
                f"{result['rho'][-1]:.12e},"
                f"{np.max(result['velocity']):.12e},"
                f"{result['velocity'][-1]:.12e},"
                f"{np.max(result['temperature']):.12e},"
                f"{result['temperature'][-1]:.12e},"
                f"{np.max(result['combined']):.12e},"
                f"{result['combined'][-1]:.12e},"
                f"{np.max(result['kinetic']):.12e},"
                f"{result['kinetic'][-1]:.12e}\n"
            )

    # ---------------------------------------------------------------
    # Maximum closure discrepancy versus scale
    # ---------------------------------------------------------------

    modes = np.array(
        [
            result["spatial_mode"]
            for result in results
        ]
    )

    epsilon = np.array(
        [
            result["epsilon"]
            for result in results
        ]
    )

    maximum_combined = np.array(
        [
            np.max(
                result["combined"]
            )
            for result in results
        ]
    )

    final_combined = np.array(
        [
            result["combined"][-1]
            for result in results
        ]
    )

    plt.figure()

    plt.loglog(
        epsilon,
        maximum_combined,
        marker="o",
        label="Maximum",
    )

    plt.loglog(
        epsilon,
        final_combined,
        marker="o",
        label="Final",
    )

    plt.xlabel(
        "Effective scale parameter epsilon"
    )

    plt.ylabel(
        "Hydrodynamic separation"
    )

    plt.title(
        "005: Closure versus spatial scale"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_005_closure_vs_scale.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Spatial scale directly
    # ---------------------------------------------------------------

    lengths = np.array(
        [
            result["L"]
            for result in results
        ]
    )

    plt.figure()

    plt.loglog(
        lengths,
        maximum_combined,
        marker="o",
        label="Maximum",
    )

    plt.loglog(
        lengths,
        final_combined,
        marker="o",
        label="Final",
    )

    plt.xlabel(
        "Characteristic spatial scale L"
    )

    plt.ylabel(
        "Hydrodynamic separation"
    )

    plt.title(
        "005: Hidden-state influence versus spatial scale"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_005_closure_vs_length.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Components
    # ---------------------------------------------------------------

    maximum_density = np.array(
        [
            np.max(result["rho"])
            for result in results
        ]
    )

    maximum_velocity = np.array(
        [
            np.max(result["velocity"])
            for result in results
        ]
    )

    maximum_temperature = np.array(
        [
            np.max(result["temperature"])
            for result in results
        ]
    )

    plt.figure()

    plt.loglog(
        epsilon,
        maximum_density,
        marker="o",
        label="Density",
    )

    plt.loglog(
        epsilon,
        maximum_velocity,
        marker="o",
        label="Velocity",
    )

    plt.loglog(
        epsilon,
        maximum_temperature,
        marker="o",
        label="Temperature",
    )

    plt.xlabel(
        "Effective scale parameter epsilon"
    )

    plt.ylabel(
        "Maximum RMS difference"
    )

    plt.title(
        "005: Components of closure breakdown"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_005_components.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Kinetic versus hydrodynamic separation
    # ---------------------------------------------------------------

    maximum_kinetic = np.array(
        [
            np.max(
                result["kinetic"]
            )
            for result in results
        ]
    )

    final_kinetic = np.array(
        [
            result["kinetic"][-1]
            for result in results
        ]
    )

    plt.figure()

    plt.loglog(
        epsilon,
        maximum_kinetic,
        marker="o",
        label="Maximum kinetic",
    )

    plt.loglog(
        epsilon,
        final_kinetic,
        marker="o",
        label="Final kinetic",
    )

    plt.xlabel(
        "Effective scale parameter epsilon"
    )

    plt.ylabel(
        "Kinetic-state separation"
    )

    plt.title(
        "005: Residual kinetic separation"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_005_kinetic_separation.png",
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
        "Experiment 005: "
        "Shrinking-Scale Closure Boundary"
    )

    print(
        "============================================"
    )

    print()
    print(
        "Results"
    )

    print(
        "-------"
    )

    for result in results:

        print(
            f"k={result['spatial_mode']:2d}, "
            f"L={result['L']:.6e}, "
            f"epsilon={result['epsilon']:.6e}, "
            f"maximum_combined="
            f"{np.max(result['combined']):.12e}, "
            f"final_combined="
            f"{result['combined'][-1]:.12e}"
        )

    print()
    print(
        "Outputs written to results/"
    )
