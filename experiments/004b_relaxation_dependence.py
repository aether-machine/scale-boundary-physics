"""
Experiment 004b: Relaxation-Time Dependence

Purpose
-------
Extend Experiment 004 by varying the kinetic relaxation time tau.

The two initial kinetic states have identical hydrodynamic moments:

    density
    velocity
    temperature

but differ in higher-order kinetic structure.

We ask whether stronger kinetic relaxation suppresses the influence
of that hidden kinetic information on subsequent hydrodynamic
variables.

The kinetic model is the same BGK-style model used in Experiment 004:

    df/dt + v df/dx = (f_eq - f) / tau

This is a controlled numerical experiment, not a derivation of
Navier-Stokes and not a simulation of a specific real fluid.
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

DT = 0.002
T_FINAL = 4.0

RELAXATION_TIMES = [
    0.05,
    0.10,
    0.20,
    0.50,
    1.00,
]

HIDDEN_AMPLITUDE = 0.35

SPATIAL_MODE = 2


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

def initial_hydrodynamic_fields(x):
    density = (
        1.0
        + 0.15
        * np.sin(
            SPATIAL_MODE * x
        )
    )

    mean_velocity = (
        0.30
        * np.cos(x)
    )

    temperature = (
        1.0
        + 0.10
        * np.sin(x)
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
    """
    Even higher-order velocity-space structure.
    """
    z = v

    return (
        (
            z ** 4
            - 6.0 * z ** 2
            + 3.0
        )
        * np.exp(
            -0.5 * z ** 2
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
    Correct candidate so that density, momentum and energy match the
    reference.

    The correction has the form:

        a + b*v + c*v^2
    """

    ref_density, ref_velocity, ref_temperature = (
        hydrodynamic_moments(
            reference,
            v,
            dv,
        )
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
):
    density, velocity, temperature = (
        initial_hydrodynamic_fields(x)
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
    density, velocity, temperature = (
        hydrodynamic_moments(
            f,
            v,
            dv,
        )
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
    """
    First-order periodic upwind streaming.
    """
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
    rho_a, u_a, t_a = (
        hydrodynamic_moments(
            f_a,
            v,
            dv,
        )
    )

    rho_b, u_b, t_b = (
        hydrodynamic_moments(
            f_b,
            v,
            dv,
        )
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
# Single relaxation-time run
# ---------------------------------------------------------------------

def run_single_tau(
    tau,
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
                tau,
            )

            f_b = kinetic_step(
                f_b,
                v,
                dx,
                dv,
                DT,
                tau,
            )

    return {
        "tau": tau,
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

    f_a, f_b = construct_initial_states(
        x,
        v,
        dv,
    )

    initial_difference = (
        hydrodynamic_difference(
            f_a,
            f_b,
            v,
            dv,
        )
    )

    print()
    print(
        "Initial hydrodynamic differences"
    )
    print(
        "---------------------------------"
    )

    print(
        f"Density RMS:     "
        f"{initial_difference[0]:.12e}"
    )

    print(
        f"Velocity RMS:    "
        f"{initial_difference[1]:.12e}"
    )

    print(
        f"Temperature RMS: "
        f"{initial_difference[2]:.12e}"
    )

    print(
        f"Combined RMS:    "
        f"{initial_difference[3]:.12e}"
    )

    results = []

    for tau in RELAXATION_TIMES:

        print()
        print(
            f"Running tau = {tau:.2f}..."
        )

        result = run_single_tau(
            tau,
            f_a,
            f_b,
            v,
            dx,
            dv,
        )

        results.append(
            result
        )

    return results, initial_difference


# ---------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------

def save_outputs(
    results,
    initial_difference,
):
    results_dir = ROOT / "results"

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------------
    # Summary CSV
    # ---------------------------------------------------------------

    csv_path = (
        results_dir
        / "experiment_004b_summary.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "tau,"
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
                f"{result['tau']:.8e},"
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
    # Summary text
    # ---------------------------------------------------------------

    summary_path = (
        results_dir
        / "experiment_004b_summary.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "Experiment 004b: "
            "Relaxation-Time Dependence\n"
        )

        handle.write(
            "========================================\n\n"
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
            "Relaxation times: "
            + ", ".join(
                f"{tau:.2f}"
                for tau in RELAXATION_TIMES
            )
            + "\n\n"
        )

        handle.write(
            "Initial hydrodynamic differences\n"
        )

        handle.write(
            "---------------------------------\n"
        )

        handle.write(
            f"Density RMS = "
            f"{initial_difference[0]:.12e}\n"
        )

        handle.write(
            f"Velocity RMS = "
            f"{initial_difference[1]:.12e}\n"
        )

        handle.write(
            f"Temperature RMS = "
            f"{initial_difference[2]:.12e}\n"
        )

        handle.write(
            f"Combined RMS = "
            f"{initial_difference[3]:.12e}\n\n"
        )

        handle.write(
            "Relaxation-time results\n"
        )

        handle.write(
            "------------------------\n"
        )

        handle.write(
            "tau,"
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
                f"{result['tau']:.8e},"
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
    # Maximum combined hydrodynamic separation
    # ---------------------------------------------------------------

    taus = [
        result["tau"]
        for result in results
    ]

    maximum_combined = [
        np.max(
            result["combined"]
        )
        for result in results
    ]

    final_combined = [
        result["combined"][-1]
        for result in results
    ]

    plt.figure()

    plt.plot(
        taus,
        maximum_combined,
        marker="o",
        label="Maximum",
    )

    plt.plot(
        taus,
        final_combined,
        marker="o",
        label="Final",
    )

    plt.xlabel(
        "Relaxation time tau"
    )

    plt.ylabel(
        "Combined hydrodynamic difference"
    )

    plt.title(
        "004b: Hydrodynamic hidden-state "
        "dependence vs relaxation time"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_004b_relaxation_dependence.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Component comparison
    # ---------------------------------------------------------------

    maximum_density = [
        np.max(result["rho"])
        for result in results
    ]

    maximum_velocity = [
        np.max(result["velocity"])
        for result in results
    ]

    maximum_temperature = [
        np.max(result["temperature"])
        for result in results
    ]

    plt.figure()

    plt.plot(
        taus,
        maximum_density,
        marker="o",
        label="Density",
    )

    plt.plot(
        taus,
        maximum_velocity,
        marker="o",
        label="Velocity",
    )

    plt.plot(
        taus,
        maximum_temperature,
        marker="o",
        label="Temperature",
    )

    plt.xlabel(
        "Relaxation time tau"
    )

    plt.ylabel(
        "Maximum RMS difference"
    )

    plt.title(
        "004b: Hydrodynamic components "
        "vs relaxation time"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_004b_components.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Time histories
    # ---------------------------------------------------------------

    plt.figure()

    for result in results:

        plt.plot(
            result["times"],
            result["combined"],
            label=f"tau={result['tau']:.2f}",
        )

    plt.xlabel("Time")

    plt.ylabel(
        "Combined hydrodynamic difference"
    )

    plt.title(
        "004b: Hidden kinetic influence "
        "over time"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_004b_time_histories.png",
        dpi=150,
    )

    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    results, initial_difference = (
        run_experiment()
    )

    save_outputs(
        results,
        initial_difference,
    )

    print()
    print(
        "Experiment 004b: "
        "Relaxation-Time Dependence"
    )

    print(
        "========================================"
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
            f"tau={result['tau']:.2f}, "
            f"maximum_combined="
            f"{np.max(result['combined']):.12e}, "
            f"final_combined="
            f"{result['combined'][-1]:.12e}, "
            f"maximum_temperature="
            f"{np.max(result['temperature']):.12e}"
        )

    print()

    print(
        "Outputs written to results/"
    )
