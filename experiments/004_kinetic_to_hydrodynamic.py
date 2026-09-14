"""
Experiment 004: Kinetic -> Hydrodynamic

Purpose
-------
Investigate whether hydrodynamic variables are dynamically closed
under a controlled kinetic model.

We construct two kinetic distributions f_A(x,v) and f_B(x,v)
that initially have the same:

    density      rho(x)
    mean velocity u(x)
    temperature  T(x)

but differ in higher-order velocity structure.

Both distributions are then evolved with a 1D BGK-style kinetic model:

    df/dt + v df/dx = (f_eq - f) / tau

The experiment asks whether the initially hidden kinetic information
can subsequently influence the hydrodynamic variables.

This is a controlled kinetic experiment. It is not a numerical
solution of the full Boltzmann equation and should not be interpreted
as a derivation of Navier-Stokes.
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

TAU = 0.20

# Strength of the higher-order kinetic perturbation.
HIDDEN_AMPLITUDE = 0.35

# Spatial modulation of the initial distribution.
SPATIAL_MODE = 2


# ---------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------

def make_grids():
    """
    Construct periodic spatial and velocity grids.
    """
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

def maxwellian(v, density, mean_velocity, temperature):
    """
    One-dimensional Maxwellian distribution.

    Parameters are arrays in x, with broadcasting over velocity.
    """
    temperature = np.maximum(
        temperature,
        1e-10,
    )

    return (
        density[:, None]
        / np.sqrt(
            2.0 * np.pi * temperature[:, None]
        )
        * np.exp(
            -(
                v[None, :] - mean_velocity[:, None]
            ) ** 2
            / (
                2.0 * temperature[:, None]
            )
        )
    )


# ---------------------------------------------------------------------
# Initial hydrodynamic fields
# ---------------------------------------------------------------------

def initial_hydrodynamic_fields(x):
    """
    Define the common initial density, velocity and temperature fields.
    """
    density = (
        1.0
        + 0.15 * np.sin(
            SPATIAL_MODE * x
        )
    )

    mean_velocity = (
        0.30 * np.cos(x)
    )

    temperature = (
        1.0
        + 0.10 * np.sin(x)
    )

    return (
        density,
        mean_velocity,
        temperature,
    )


# ---------------------------------------------------------------------
# Hidden kinetic perturbation
# ---------------------------------------------------------------------

def hidden_velocity_shape(v):
    """
    Construct an even velocity-space perturbation.

    The perturbation is designed to alter higher velocity moments while
    being approximately orthogonal to the first three hydrodynamic
    moments.

    A Hermite-like polynomial is used:

        H4(z) = z^4 - 6 z^2 + 3

    multiplied by a Gaussian envelope.

    This is not intended as an exact orthogonal projection under every
    finite-grid discretization. The hydrodynamic moments are explicitly
    corrected after construction.
    """
    z = v

    return (
        (z ** 4 - 6.0 * z ** 2 + 3.0)
        * np.exp(
            -0.5 * z ** 2
        )
    )


# ---------------------------------------------------------------------
# Construct A and B
# ---------------------------------------------------------------------

def construct_initial_states(
    x,
    v,
    dv,
):
    """
    Construct two kinetic states with the same hydrodynamic moments.

    State A is a local Maxwellian.

    State B is A plus a higher-order velocity perturbation.

    After adding the perturbation, B is projected back onto the
    subspace that preserves density, momentum and energy.
    """
    density, mean_velocity, temperature = (
        initial_hydrodynamic_fields(x)
    )

    f_a = maxwellian(
        v,
        density,
        mean_velocity,
        temperature,
    )

    shape = hidden_velocity_shape(v)

    # Local amplitude follows the spatial density so that the hidden
    # state is genuinely spatially distributed.
    perturbation = (
        HIDDEN_AMPLITUDE
        * density[:, None]
        * shape[None, :]
        * np.exp(
            -0.5
            * (
                v[None, :]
                - mean_velocity[:, None]
            ) ** 2
            / temperature[:, None]
        )
    )

    f_b = f_a + perturbation

    # Prevent negative values before the moment correction.
    f_b = np.maximum(
        f_b,
        1e-12,
    )

    # Project B so that its density, momentum and energy match A.
    f_b = match_hydrodynamic_moments(
        f_a,
        f_b,
        v,
        dv,
    )

    # Positivity correction after projection.
    f_b = np.maximum(
        f_b,
        1e-12,
    )

    # One final moment matching pass.
    f_b = match_hydrodynamic_moments(
        f_a,
        f_b,
        v,
        dv,
    )

    return f_a, f_b


# ---------------------------------------------------------------------
# Moment matching
# ---------------------------------------------------------------------

def hydrodynamic_moments(
    f,
    v,
    dv,
):
    """
    Calculate density, mean velocity and temperature.
    """
    density = np.sum(
        f,
        axis=1,
    ) * dv

    momentum = np.sum(
        f * v[None, :],
        axis=1,
    ) * dv

    mean_velocity = (
        momentum
        / np.maximum(
            density,
            1e-14,
        )
    )

    energy = np.sum(
        f
        * v[None, :] ** 2,
        axis=1,
    ) * dv

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


def match_hydrodynamic_moments(
    reference,
    candidate,
    v,
    dv,
):
    """
    Correct candidate so that its density, momentum and energy match
    the reference distribution.

    The correction is constructed as:

        candidate + a + b*v + c*v^2

    with coefficients chosen independently at each spatial position.
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

    current_density, current_velocity, current_temperature = (
        hydrodynamic_moments(
            candidate,
            v,
            dv,
        )
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

    # Moment matrix.
    moments = np.array(
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
        moments
    )

    corrected = candidate.copy()

    for i in range(candidate.shape[0]):

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
                target - current
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
# Kinetic equilibrium
# ---------------------------------------------------------------------

def equilibrium_from_moments(
    f,
    v,
    dv,
):
    """
    Construct the local Maxwellian having the same hydrodynamic
    moments as f.
    """
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
# Spatial streaming
# ---------------------------------------------------------------------

def streaming_step(
    f,
    x,
    v,
    dx,
    dt,
):
    """
    First-order periodic upwind streaming.

    The equation is:

        df/dt + v df/dx = 0

    Positive velocities use a backward difference.
    Negative velocities use a forward difference.
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
# Kinetic time step
# ---------------------------------------------------------------------

def kinetic_step(
    f,
    x,
    v,
    dx,
    dv,
    dt,
):
    """
    One operator-split kinetic timestep.

    1. Streaming
    2. BGK relaxation
    """
    streamed = streaming_step(
        f,
        x,
        v,
        dx,
        dt,
    )

    equilibrium = equilibrium_from_moments(
        streamed,
        v,
        dv,
    )

    relaxation_factor = np.exp(
        -dt / TAU
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
    """
    Return RMS differences in rho, u and T.
    """
    rho_a, u_a, t_a = hydrodynamic_moments(
        f_a,
        v,
        dv,
    )

    rho_b, u_b, t_b = hydrodynamic_moments(
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
    """
    L2 difference between kinetic distributions.
    """
    return np.sqrt(
        np.sum(
            (f_a - f_b) ** 2
        )
        * dx
        * dv
    )


# ---------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------

def run_experiment():
    x, v, dx, dv = make_grids()

    f_a, f_b = construct_initial_states(
        x,
        v,
        dv,
    )

    initial_hydro_difference = (
        hydrodynamic_difference(
            f_a,
            f_b,
            v,
            dv,
        )
    )

    initial_kinetic_difference = (
        kinetic_difference(
            f_a,
            f_b,
            dx,
            dv,
        )
    )

    times = []
    kinetic_history = []

    rho_history = []
    velocity_history = []
    temperature_history = []
    combined_history = []

    n_steps = int(
        round(
            T_FINAL / DT
        )
    )

    sample_interval = max(
        1,
        n_steps // 500,
    )

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

            kinetic_history.append(
                kinetic_difference(
                    f_a,
                    f_b,
                    dx,
                    dv,
                )
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

        if step < n_steps:

            f_a = kinetic_step(
                f_a,
                x,
                v,
                dx,
                dv,
                DT,
            )

            f_b = kinetic_step(
                f_b,
                x,
                v,
                dx,
                dv,
                DT,
            )

    return {
        "x": x,
        "v": v,
        "f_a": f_a,
        "f_b": f_b,
        "times": np.asarray(times),
        "kinetic": np.asarray(
            kinetic_history
        ),
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
        "initial_hydro": (
            initial_hydro_difference
        ),
        "initial_kinetic": (
            initial_kinetic_difference
        ),
    }


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------

def save_outputs(result):
    results_dir = ROOT / "results"

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    times = result["times"]

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    summary_path = (
        results_dir
        / "experiment_004_summary.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "Experiment 004: "
            "Kinetic -> Hydrodynamic\n"
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
            f"Relaxation time tau: {TAU}\n\n"
        )

        handle.write(
            "Initial-state conditions\n"
        )

        handle.write(
            "------------------------\n"
        )

        handle.write(
            "The two kinetic states differ,\n"
        )

        handle.write(
            "but density, velocity and temperature "
            "are matched initially.\n\n"
        )

        (
            initial_rho,
            initial_velocity,
            initial_temperature,
            initial_combined,
        ) = result["initial_hydro"]

        handle.write(
            "Initial hydrodynamic differences\n"
        )

        handle.write(
            f"density RMS = "
            f"{initial_rho:.12e}\n"
        )

        handle.write(
            f"velocity RMS = "
            f"{initial_velocity:.12e}\n"
        )

        handle.write(
            f"temperature RMS = "
            f"{initial_temperature:.12e}\n"
        )

        handle.write(
            f"combined RMS = "
            f"{initial_combined:.12e}\n\n"
        )

        handle.write(
            "Initial kinetic L2 difference = "
            f"{result['initial_kinetic']:.12e}\n\n"
        )

        handle.write(
            "Evolution\n"
        )

        handle.write(
            "---------\n"
        )

        handle.write(
            f"Maximum density difference = "
            f"{np.max(result['rho']):.12e}\n"
        )

        handle.write(
            f"Final density difference = "
            f"{result['rho'][-1]:.12e}\n"
        )

        handle.write(
            f"Maximum velocity difference = "
            f"{np.max(result['velocity']):.12e}\n"
        )

        handle.write(
            f"Final velocity difference = "
            f"{result['velocity'][-1]:.12e}\n"
        )

        handle.write(
            f"Maximum temperature difference = "
            f"{np.max(result['temperature']):.12e}\n"
        )

        handle.write(
            f"Final temperature difference = "
            f"{result['temperature'][-1]:.12e}\n"
        )

        handle.write(
            f"Maximum combined hydrodynamic difference = "
            f"{np.max(result['combined']):.12e}\n"
        )

        handle.write(
            f"Final combined hydrodynamic difference = "
            f"{result['combined'][-1]:.12e}\n"
        )

        handle.write(
            f"Maximum kinetic L2 difference = "
            f"{np.max(result['kinetic']):.12e}\n"
        )

        handle.write(
            f"Final kinetic L2 difference = "
            f"{result['kinetic'][-1]:.12e}\n"
        )

    # ---------------------------------------------------------------
    # Hydrodynamic separation plot
    # ---------------------------------------------------------------

    plt.figure()

    plt.plot(
        times,
        result["rho"],
        label="Density",
    )

    plt.plot(
        times,
        result["velocity"],
        label="Velocity",
    )

    plt.plot(
        times,
        result["temperature"],
        label="Temperature",
    )

    plt.xlabel("Time")
    plt.ylabel("RMS difference")
    plt.title(
        "004: Hidden kinetic information "
        "and hydrodynamic separation"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_004_hydrodynamic_separation.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Combined separation
    # ---------------------------------------------------------------

    plt.figure()

    plt.plot(
        times,
        result["combined"],
    )

    plt.xlabel("Time")
    plt.ylabel(
        "Combined hydrodynamic difference"
    )
    plt.title(
        "004: Combined hydrodynamic separation"
    )
    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_004_combined_separation.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Kinetic separation
    # ---------------------------------------------------------------

    plt.figure()

    plt.plot(
        times,
        result["kinetic"],
    )

    plt.xlabel("Time")
    plt.ylabel(
        "Kinetic L2 difference"
    )
    plt.title(
        "004: Kinetic-state separation"
    )
    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_004_kinetic_separation.png",
        dpi=150,
    )

    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":

    result = run_experiment()

    save_outputs(result)

    print()
    print(
        "Experiment 004: "
        "Kinetic -> Hydrodynamic"
    )
    print(
        "========================================"
    )
    print()

    print(
        "Initial hydrodynamic differences"
    )
    print(
        "---------------------------------"
    )

    (
        rho,
        velocity,
        temperature,
        combined,
    ) = result["initial_hydro"]

    print(
        f"Density RMS:      "
        f"{rho:.12e}"
    )

    print(
        f"Velocity RMS:     "
        f"{velocity:.12e}"
    )

    print(
        f"Temperature RMS:  "
        f"{temperature:.12e}"
    )

    print(
        f"Combined RMS:     "
        f"{combined:.12e}"
    )

    print()

    print(
        "Initial kinetic L2 difference:"
    )

    print(
        f"{result['initial_kinetic']:.12e}"
    )

    print()

    print(
        "Evolution"
    )

    print(
        "---------"
    )

    print(
        f"Maximum density difference: "
        f"{np.max(result['rho']):.12e}"
    )

    print(
        f"Final density difference:   "
        f"{result['rho'][-1]:.12e}"
    )

    print(
        f"Maximum velocity difference: "
        f"{np.max(result['velocity']):.12e}"
    )

    print(
        f"Final velocity difference:   "
        f"{result['velocity'][-1]:.12e}"
    )

    print(
        f"Maximum temperature difference: "
        f"{np.max(result['temperature']):.12e}"
    )

    print(
        f"Final temperature difference:   "
        f"{result['temperature'][-1]:.12e}"
    )

    print(
        f"Maximum combined difference: "
        f"{np.max(result['combined']):.12e}"
    )

    print(
        f"Final combined difference:   "
        f"{result['combined'][-1]:.12e}"
    )

    print()

    print(
        "Outputs written to results/"
    )
