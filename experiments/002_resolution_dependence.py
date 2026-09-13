"""
Experiment 002: Resolution dependence.

This experiment investigates how the discrepancy between microscopic
and effective dynamics depends on:

1. The amplitude of unresolved high-frequency structure.
2. The coarse-graining block size.

The underlying microscopic and effective equations are kept fixed.

For each parameter combination we calculate

    D_max = max_t Delta(t)

where

    Delta(t) =
        C[Phi_micro(t, X0)]
        -
        Phi_eff(t, C[X0])

The purpose is to determine whether the discrepancy depends
systematically on the resolution of the representation.

This is still a methodological experiment. The effective model is
an assumed closure and is not derived rigorously from the microscopic
model.
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

HIGH_FREQUENCY_AMPLITUDES = [
    0.0,
    0.05,
    0.10,
    0.20,
]

BLOCK_SIZES = [
    2,
    4,
    8,
    16,
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

def initial_condition(n, high_frequency_amplitude):
    """
    Construct the microscopic initial condition.

    A low-frequency mode provides the large-scale structure.

    A high-frequency perturbation represents structure that may be
    partially or completely lost during coarse-graining.
    """

    x = np.arange(n)

    low_frequency = AMPLITUDE * np.sin(
        2.0 * np.pi * WAVELENGTHS * x / n
    )

    high_frequency = high_frequency_amplitude * np.sin(
        2.0 * np.pi * (WAVELENGTHS * 8) * x / n
    )

    q0 = low_frequency + high_frequency
    p0 = np.zeros_like(q0)

    return q0, p0


# ---------------------------------------------------------------------------
# Single simulation
# ---------------------------------------------------------------------------

def run_single_case(block_size, high_frequency_amplitude):
    """
    Run one parameter combination.

    Returns the maximum combined discrepancy.
    """

    q_micro, p_micro = initial_condition(
        N_MICRO,
        high_frequency_amplitude,
    )

    q_effective, p_effective = coarse_grain(
        q_micro,
        p_micro,
        block_size,
    )

    n_steps = int(round(T_FINAL / DT))

    maximum_discrepancy = 0.0

    for _ in range(n_steps):

        q_micro, p_micro = micro_rk4_step(
            q_micro,
            p_micro,
            DT,
        )

        q_effective, p_effective = effective_rk4_step(
            q_effective,
            p_effective,
            DT,
            k=K,
            omega0_sq=OMEGA0_SQ,
            beta=BETA,
        )

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

    return maximum_discrepancy


# ---------------------------------------------------------------------------
# Main parameter sweep
# ---------------------------------------------------------------------------

def run_experiment():
    """Run the complete resolution-dependence parameter sweep."""

    results = []

    for block_size in BLOCK_SIZES:

        print(f"Block size: {block_size}")

        for high_frequency_amplitude in HIGH_FREQUENCY_AMPLITUDES:

            print(
                f"  high-frequency amplitude: "
                f"{high_frequency_amplitude:.2f}"
            )

            maximum_discrepancy = run_single_case(
                block_size,
                high_frequency_amplitude,
            )

            results.append(
                {
                    "block_size": block_size,
                    "high_frequency_amplitude":
                        high_frequency_amplitude,
                    "maximum_discrepancy":
                        maximum_discrepancy,
                }
            )

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_outputs(results):
    """Save numerical results, plots, and a summary."""

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------------
    # Numerical table
    # ---------------------------------------------------------------

    table_path = results_dir / "experiment_002_results.csv"

    with table_path.open("w", encoding="utf-8") as f:

        f.write(
            "block_size,"
            "high_frequency_amplitude,"
            "maximum_discrepancy\n"
        )

        for result in results:

            f.write(
                f"{result['block_size']},"
                f"{result['high_frequency_amplitude']:.6f},"
                f"{result['maximum_discrepancy']:.10e}\n"
            )

    # ---------------------------------------------------------------
    # Plot 1: discrepancy versus unresolved amplitude
    # ---------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for block_size in BLOCK_SIZES:

        amplitudes = []
        discrepancies = []

        for result in results:

            if result["block_size"] == block_size:

                amplitudes.append(
                    result["high_frequency_amplitude"]
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

    plt.xlabel("High-frequency amplitude")
    plt.ylabel("Maximum discrepancy")
    plt.title(
        "Experiment 002: Discrepancy vs Unresolved Structure"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "experiment_002_amplitude.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Plot 2: discrepancy versus block size
    # ---------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for amplitude in HIGH_FREQUENCY_AMPLITUDES:

        block_sizes = []
        discrepancies = []

        for result in results:

            if result["high_frequency_amplitude"] == amplitude:

                block_sizes.append(
                    result["block_size"]
                )

                discrepancies.append(
                    result["maximum_discrepancy"]
                )

        plt.plot(
            block_sizes,
            discrepancies,
            marker="o",
            label=f"Amplitude {amplitude:.2f}",
        )

    plt.xlabel("Coarse-graining block size")
    plt.ylabel("Maximum discrepancy")
    plt.title(
        "Experiment 002: Discrepancy vs Resolution"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "experiment_002_block_size.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Text summary
    # ---------------------------------------------------------------

    summary_path = (
        results_dir / "experiment_002_summary.txt"
    )

    with summary_path.open("w", encoding="utf-8") as f:

        f.write(
            "Experiment 002: Resolution Dependence\n"
        )

        f.write(
            "======================================\n\n"
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
            "High-frequency amplitudes:\n"
        )

        f.write(
            f"{HIGH_FREQUENCY_AMPLITUDES}\n\n"
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
                f"high_frequency="
                f"{result['high_frequency_amplitude']:.2f}, "
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
    print("Experiment 002 complete.")
    print("Results written to: results/")
