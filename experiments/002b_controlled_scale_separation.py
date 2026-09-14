"""
Experiment 002b: Controlled scale separation.

This experiment investigates whether the discrepancy between
microscopic and effective dynamics depends systematically on the
relationship between hidden structure and coarse-graining scale.

Three parameters are varied:

1. Coarse-graining block size B.
2. Hidden structure wavelength lambda_h.
3. Hidden structure amplitude A_h.

The underlying microscopic and effective equations remain unchanged.

For each parameter combination we calculate

    D_max = max_t Delta(t)

where

    Delta(t) =
        C[Phi_micro(t, X0)]
        -
        Phi_eff(t, C[X0])

We also record the scale ratio

    R = lambda_h / B

The purpose is to determine whether the discrepancy depends primarily
on the relationship between the characteristic hidden scale and the
resolution of the effective representation.

This remains a methodological experiment. The effective model is an
assumed closure and is not derived rigorously from the microscopic
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
    0.40,
]

HIDDEN_WAVELENGTHS = [
    4,
    8,
    16,
    32,
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

def initial_condition(n, hidden_amplitude, hidden_wavelength):
    """
    Construct the microscopic initial condition.

    A low-frequency mode provides the resolved large-scale structure.

    A second mode provides hidden structure with independently
    controlled amplitude and wavelength.
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
    hidden_wavelength,
):
    """
    Run one parameter combination.

    Returns the maximum combined discrepancy.
    """

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
# Parameter sweep
# ---------------------------------------------------------------------------

def run_experiment():
    """Run the complete controlled scale-separation sweep."""

    results = []

    total_cases = (
        len(BLOCK_SIZES)
        * len(HIGH_FREQUENCY_AMPLITUDES)
        * len(HIDDEN_WAVELENGTHS)
    )

    case_number = 0

    for block_size in BLOCK_SIZES:

        for hidden_wavelength in HIDDEN_WAVELENGTHS:

            for hidden_amplitude in HIGH_FREQUENCY_AMPLITUDES:

                case_number += 1

                maximum_discrepancy = run_single_case(
                    block_size,
                    hidden_amplitude,
                    hidden_wavelength,
                )

                scale_ratio = (
                    hidden_wavelength / block_size
                )

                results.append(
                    {
                        "block_size": block_size,
                        "hidden_wavelength":
                            hidden_wavelength,
                        "hidden_amplitude":
                            hidden_amplitude,
                        "scale_ratio":
                            scale_ratio,
                        "maximum_discrepancy":
                            maximum_discrepancy,
                    }
                )

                print(
                    f"[{case_number}/{total_cases}] "
                    f"B={block_size:2d}, "
                    f"lambda={hidden_wavelength:2d}, "
                    f"A={hidden_amplitude:.2f}, "
                    f"R={scale_ratio:.2f}, "
                    f"Dmax={maximum_discrepancy:.6e}"
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
    # Numerical table
    # ---------------------------------------------------------------

    table_path = (
        results_dir / "experiment_002b_results.csv"
    )

    with table_path.open("w", encoding="utf-8") as f:

        f.write(
            "block_size,"
            "hidden_wavelength,"
            "hidden_amplitude,"
            "scale_ratio,"
            "maximum_discrepancy\n"
        )

        for result in results:

            f.write(
                f"{result['block_size']},"
                f"{result['hidden_wavelength']},"
                f"{result['hidden_amplitude']:.6f},"
                f"{result['scale_ratio']:.6f},"
                f"{result['maximum_discrepancy']:.10e}\n"
            )

    # ---------------------------------------------------------------
    # Plot 1: discrepancy versus hidden wavelength
    # ---------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for block_size in BLOCK_SIZES:

        for hidden_amplitude in [0.10]:

            wavelengths = []
            discrepancies = []

            for result in results:

                if (
                    result["block_size"] == block_size
                    and result["hidden_amplitude"]
                    == hidden_amplitude
                ):

                    wavelengths.append(
                        result["hidden_wavelength"]
                    )

                    discrepancies.append(
                        result["maximum_discrepancy"]
                    )

            plt.plot(
                wavelengths,
                discrepancies,
                marker="o",
                label=f"Block size {block_size}",
            )

    plt.xlabel("Hidden wavelength")
    plt.ylabel("Maximum discrepancy")
    plt.title(
        "Experiment 002b: Discrepancy vs Hidden Wavelength"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "experiment_002b_wavelength.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Plot 2: discrepancy versus hidden amplitude
    # ---------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for block_size in BLOCK_SIZES:

        for hidden_wavelength in [8]:

            amplitudes = []
            discrepancies = []

            for result in results:

                if (
                    result["block_size"] == block_size
                    and result["hidden_wavelength"]
                    == hidden_wavelength
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
        "Experiment 002b: Discrepancy vs Hidden Amplitude"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "experiment_002b_amplitude.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Plot 3: discrepancy versus scale ratio
    # ---------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for hidden_amplitude in [0.05, 0.10, 0.20]:

        ratios = []
        discrepancies = []

        for result in results:

            if (
                result["hidden_amplitude"]
                == hidden_amplitude
            ):

                ratios.append(
                    result["scale_ratio"]
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
        "Experiment 002b: Discrepancy vs Scale Ratio"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "experiment_002b_scale_ratio.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Text summary
    # ---------------------------------------------------------------

    summary_path = (
        results_dir / "experiment_002b_summary.txt"
    )

    with summary_path.open("w", encoding="utf-8") as f:

        f.write(
            "Experiment 002b: Controlled Scale Separation\n"
        )

        f.write(
            "==============================================\n\n"
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
            f"{HIGH_FREQUENCY_AMPLITUDES}\n\n"
        )

        f.write(
            "Hidden wavelengths:\n"
        )

        f.write(
            f"{HIDDEN_WAVELENGTHS}\n\n"
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
                f"amplitude="
                f"{result['hidden_amplitude']:.2f}, "
                f"ratio="
                f"{result['scale_ratio']:.3f}, "
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
    print("Experiment 002b complete.")
    print("Results written to: results/")
