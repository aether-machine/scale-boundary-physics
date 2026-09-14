"""
Experiment 002c: Resolution Boundary.

This experiment asks whether the discrepancy between microscopic
evolution followed by coarse-graining and independently evolved
effective dynamics changes systematically when a spatial mode crosses
the resolution scale of the coarse representation.

For a coarse-graining block size B, three hidden wavelengths are used:

    lambda = 4B
        clearly resolved

    lambda = B
        boundary-scale

    lambda = B/2
        sub-resolution

The hidden amplitude is also varied.

The central quantity is

    D_max = max_t Delta(t)

where

    Delta(t) =
        C[Phi_micro(t, X0)]
        -
        Phi_eff(t, C[X0])

This is a methodological experiment. The effective model is an
assumed closure and is not derived rigorously from the microscopic
model.

The experiment therefore tests for systematic scale-dependent
behaviour, not for a physical singularity or a proof of the
scale-boundary hypothesis.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Repository setup
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from scale_boundary.coarse_grain import coarse_grain
from scale_boundary.effective import rk4_step as effective_rk4_step
from scale_boundary.metrics import state_discrepancy
from scale_boundary.micro import rhs as micro_rhs


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

N_MICRO = 128

DT = 0.002
T_FINAL = 20.0

K = 1.0
OMEGA0_SQ = 0.5
BETA = 0.25

AMPLITUDE = 0.8
WAVELENGTHS = 4

HIDDEN_AMPLITUDES = [
    0.0,
    0.05,
    0.10,
    0.20,
    0.40,
]

BLOCK_SIZES = [
    2,
    4,
    8,
    16,
]

# Relative hidden wavelengths.
#
# 4.0 -> clearly resolved
# 1.0 -> boundary scale
# 0.5 -> sub-resolution

RELATIVE_WAVELENGTHS = [
    4.0,
    1.0,
    0.5,
]


# ---------------------------------------------------------------------------
# Numerical integration
# ---------------------------------------------------------------------------

def micro_rk4_step(q, p, dt):
    """Advance the microscopic system by one RK4 step."""

    k1_q, k1_p = micro_rhs(
        q,
        p,
        k=K,
        omega0_sq=OMEGA0_SQ,
        beta=BETA,
    )

    k2_q, k2_p = micro_rhs(
        q + 0.5 * dt * k1_q,
        p + 0.5 * dt * k1_p,
        k=K,
        omega0_sq=OMEGA0_SQ,
        beta=BETA,
    )

    k3_q, k3_p = micro_rhs(
        q + 0.5 * dt * k2_q,
        p + 0.5 * dt * k2_p,
        k=K,
        omega0_sq=OMEGA0_SQ,
        beta=BETA,
    )

    k4_q, k4_p = micro_rhs(
        q + dt * k3_q,
        p + dt * k3_p,
        k=K,
        omega0_sq=OMEGA0_SQ,
        beta=BETA,
    )

    q_next = q + (dt / 6.0) * (
        k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q
    )

    p_next = p + (dt / 6.0) * (
        k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p
    )

    return q_next, p_next


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------

def initial_condition(
    n,
    hidden_amplitude,
    hidden_wavelength,
):
    """
    Construct the microscopic initial condition.

    A low-frequency mode provides the resolved large-scale structure.

    A second mode provides the controlled test structure whose
    wavelength is defined relative to the coarse-graining block size.
    """

    x = np.arange(n)

    low_frequency = AMPLITUDE * np.sin(
        2.0 * np.pi * WAVELENGTHS * x / n
    )

    hidden_structure = hidden_amplitude * np.sin(
        2.0 * np.pi * x / hidden_wavelength
    )

    q0 = low_frequency + hidden_structure
    p0 = np.zeros_like(q0)

    return q0, p0


# ---------------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------------

def run_single_case(
    block_size,
    hidden_amplitude,
    relative_wavelength,
):
    """
    Run one parameter combination.

    Returns the maximum combined discrepancy.
    """

    hidden_wavelength = (
        block_size * relative_wavelength
    )

    # Require an integer wavelength compatible with the
    # microscopic lattice.
    if not hidden_wavelength.is_integer():
        raise ValueError(
            "Hidden wavelength must be an integer "
            f"for block size {block_size} and "
            f"relative wavelength {relative_wavelength}."
        )

    hidden_wavelength = int(hidden_wavelength)

    if N_MICRO % hidden_wavelength != 0:
        raise ValueError(
            f"Hidden wavelength {hidden_wavelength} "
            "must divide the microscopic lattice size."
        )

    q_micro, p_micro = initial_condition(
        N_MICRO,
        hidden_amplitude,
        hidden_wavelength,
    )

    q_effective, p_effective = coarse_grain(
        q_micro,
        p_micro,
        block_size,
    )

    n_steps = int(round(T_FINAL / DT))

    maximum_discrepancy = 0.0

    for _ in range(n_steps):

        # Route A:
        # evolve microscopic state first.
        q_micro, p_micro = micro_rk4_step(
            q_micro,
            p_micro,
            DT,
        )

        # Route B:
        # evolve coarse state independently.
        q_effective, p_effective = effective_rk4_step(
            q_effective,
            p_effective,
            DT,
            k=K,
            omega0_sq=OMEGA0_SQ,
            beta=BETA,
        )

        # Coarse-grain the microscopic state.
        q_micro_coarse, p_micro_coarse = coarse_grain(
            q_micro,
            p_micro,
            block_size,
        )

        _, _, combined = state_discrepancy(
            q_micro_coarse,
            p_micro_coarse,
            q_effective,
            p_effective,
        )

        maximum_discrepancy = max(
            maximum_discrepancy,
            combined,
        )

    return (
        hidden_wavelength,
        maximum_discrepancy,
    )


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def run_experiment():
    """Run the complete resolution-boundary experiment."""

    results = []

    total_cases = (
        len(BLOCK_SIZES)
        * len(HIDDEN_AMPLITUDES)
        * len(RELATIVE_WAVELENGTHS)
    )

    case_number = 0

    for block_size in BLOCK_SIZES:

        for relative_wavelength in RELATIVE_WAVELENGTHS:

            for hidden_amplitude in HIDDEN_AMPLITUDES:

                case_number += 1

                (
                    hidden_wavelength,
                    maximum_discrepancy,
                ) = run_single_case(
                    block_size,
                    hidden_amplitude,
                    relative_wavelength,
                )

                results.append(
                    {
                        "block_size": block_size,
                        "relative_wavelength":
                            relative_wavelength,
                        "hidden_wavelength":
                            hidden_wavelength,
                        "hidden_amplitude":
                            hidden_amplitude,
                        "maximum_discrepancy":
                            maximum_discrepancy,
                    }
                )

                print(
                    f"[{case_number}/{total_cases}] "
                    f"B={block_size:2d}, "
                    f"lambda={hidden_wavelength:2d}, "
                    f"lambda/B="
                    f"{relative_wavelength:.2f}, "
                    f"A={hidden_amplitude:.2f}, "
                    f"Dmax="
                    f"{maximum_discrepancy:.6e}"
                )

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_outputs(results):
    """Save numerical results, plots, and summary."""

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------------
    # CSV
    # ---------------------------------------------------------------

    csv_path = (
        results_dir
        / "experiment_002c_results.csv"
    )

    with csv_path.open("w", encoding="utf-8") as f:

        f.write(
            "block_size,"
            "relative_wavelength,"
            "hidden_wavelength,"
            "hidden_amplitude,"
            "maximum_discrepancy\n"
        )

        for result in results:

            f.write(
                f"{result['block_size']},"
                f"{result['relative_wavelength']:.2f},"
                f"{result['hidden_wavelength']},"
                f"{result['hidden_amplitude']:.6f},"
                f"{result['maximum_discrepancy']:.10e}\n"
            )

    # ---------------------------------------------------------------
    # Plot 1: resolution regime
    # ---------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for block_size in BLOCK_SIZES:

        amplitudes = []
        discrepancies = []

        for result in results:

            if (
                result["block_size"] == block_size
                and result["relative_wavelength"] == 1.0
            ):

                amplitudes.append(
                    result["hidden_amplitude"]
                )

                discrepancies.append(
                    result["maximum_discrepancy"]
                )

        plt.plot(
            amplitudes,
            discrepancies,
            marker="o",
            label=f"Block size {block_size}",
        )

    plt.xlabel("Hidden structure amplitude")
    plt.ylabel("Maximum discrepancy")
    plt.title(
        "Experiment 002c: Boundary-Scale Structure"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_002c_boundary.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Plot 2: resolved / boundary / sub-resolution
    # ---------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    regime_labels = [
        "sub-resolution\nlambda/B = 0.5",
        "boundary\nlambda/B = 1.0",
        "resolved\nlambda/B = 4.0",
    ]

    regime_values = [0.5, 1.0, 4.0]

    for block_size in BLOCK_SIZES:

        values = []

        for ratio in regime_values:

            matching = [
                result["maximum_discrepancy"]
                for result in results
                if (
                    result["block_size"] == block_size
                    and result["relative_wavelength"]
                    == ratio
                    and result["hidden_amplitude"]
                    == 0.20
                )
            ]

            values.append(matching[0])

        plt.plot(
            regime_labels,
            values,
            marker="o",
            label=f"Block size {block_size}",
        )

    plt.xlabel("Resolution regime")
    plt.ylabel("Maximum discrepancy")
    plt.title(
        "Experiment 002c: Crossing the Resolution Boundary"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_002c_regimes.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Plot 3: discrepancy versus lambda/B
    # ---------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for hidden_amplitude in [0.05, 0.10, 0.20, 0.40]:

        ratios = []
        discrepancies = []

        for result in results:

            if (
                result["hidden_amplitude"]
                == hidden_amplitude
            ):

                ratios.append(
                    result["relative_wavelength"]
                )

                discrepancies.append(
                    result["maximum_discrepancy"]
                )

        plt.scatter(
            ratios,
            discrepancies,
            label=f"Amplitude {hidden_amplitude:.2f}",
        )

    plt.xlabel("Hidden wavelength / block size")
    plt.ylabel("Maximum discrepancy")
    plt.title(
        "Experiment 002c: Scale Ratio"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_002c_scale_ratio.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Text summary
    # ---------------------------------------------------------------

    summary_path = (
        results_dir
        / "experiment_002c_summary.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Experiment 002c: Resolution Boundary\n"
        )

        f.write(
            "=====================================\n\n"
        )

        f.write(
            f"Microscopic sites: {N_MICRO}\n"
        )

        f.write(
            f"Time step: {DT}\n"
        )

        f.write(
            f"Final time: {T_FINAL}\n\n"
        )

        f.write(
            "Hidden amplitudes:\n"
        )

        f.write(
            f"{HIDDEN_AMPLITUDES}\n\n"
        )

        f.write(
            "Relative wavelengths (lambda/B):\n"
        )

        f.write(
            f"{RELATIVE_WAVELENGTHS}\n\n"
        )

        f.write(
            "Block sizes:\n"
        )

        f.write(
            f"{BLOCK_SIZES}\n\n"
        )

        f.write(
            "Results\n"
            "-------\n"
        )

        for result in results:

            f.write(
                f"block={result['block_size']:2d}, "
                f"wavelength="
                f"{result['hidden_wavelength']:2d}, "
                f"lambda/B="
                f"{result['relative_wavelength']:.2f}, "
                f"amplitude="
                f"{result['hidden_amplitude']:.2f}, "
                f"maximum_discrepancy="
                f"{result['maximum_discrepancy']:.10e}\n"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    results = run_experiment()

    save_outputs(results)

    print()
    print("Experiment 002c complete.")
    print("Results written to: results/")
