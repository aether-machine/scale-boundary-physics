"""
Experiment 001: Microscopic dynamics versus effective dynamics.

This experiment compares two routes from the same initial condition:

Route A:
    X0 -> microscopic evolution -> coarse-graining

Route B:
    X0 -> coarse-graining -> effective evolution

The discrepancy between the two routes is

    Delta(t) =
        C[Phi_micro(t, X0)]
        -
        Phi_eff(t, C[X0])

The purpose of this first experiment is methodological. It tests
whether the computational framework can measure a difference between
microscopic evolution and independently evolved coarse dynamics.

The effective model used here is deliberately simple and is NOT
derived from the microscopic model. That distinction is important:
this experiment establishes the measurement framework, not a claim
about the correct coarse-grained physics.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


# Allow the experiment to be run directly from the repository root
# without requiring the package to be installed first.

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from scale_boundary.coarse_grain import coarse_grain
from scale_boundary.effective import rk4_step as effective_rk4_step
from scale_boundary.metrics import state_discrepancy, summarize_discrepancy
from scale_boundary.micro import rhs as micro_rhs


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

N_MICRO = 128
BLOCK_SIZE = 8
N_COARSE = N_MICRO // BLOCK_SIZE

DT = 0.002
T_FINAL = 20.0

K = 1.0
OMEGA0_SQ = 0.5
BETA = 0.25

AMPLITUDE = 0.8
WAVELENGTHS = 4


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

def initial_condition(n):
    """
    Construct the microscopic initial condition.

    A low-frequency mode provides the large-scale structure.
    A smaller high-frequency perturbation introduces unresolved
    structure within the coarse blocks.
    """

    x = np.arange(n)

    low_frequency = AMPLITUDE * np.sin(
        2.0 * np.pi * WAVELENGTHS * x / n
    )

    high_frequency = 0.1 * np.sin(
        2.0 * np.pi * (WAVELENGTHS * 8) * x / n
    )

    q0 = low_frequency + high_frequency
    p0 = np.zeros_like(q0)

    return q0, p0


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment():
    """Run Experiment 001 and return recorded results."""

    q_micro, p_micro = initial_condition(N_MICRO)

    # Route B begins by coarse-graining the initial microscopic state.
    q_effective, p_effective = coarse_grain(
        q_micro,
        p_micro,
        BLOCK_SIZE,
    )

    initial_q_effective = q_effective.copy()
    initial_p_effective = p_effective.copy()

    n_steps = int(round(T_FINAL / DT))

    times = np.empty(n_steps + 1)

    q_discrepancy = np.empty(n_steps + 1)
    p_discrepancy = np.empty(n_steps + 1)
    combined_discrepancy = np.empty(n_steps + 1)

    micro_coarse_history = []
    effective_history = []

    # At t = 0 both routes are identical by construction.
    q_micro_coarse, p_micro_coarse = coarse_grain(
        q_micro,
        p_micro,
        BLOCK_SIZE,
    )

    dq, dp, dcombined = state_discrepancy(
        q_micro_coarse,
        p_micro_coarse,
        q_effective,
        p_effective,
    )

    times[0] = 0.0
    q_discrepancy[0] = dq
    p_discrepancy[0] = dp
    combined_discrepancy[0] = dcombined

    micro_coarse_history.append(q_micro_coarse.copy())
    effective_history.append(q_effective.copy())

    for step in range(1, n_steps + 1):

        # Route A:
        # evolve the microscopic state, then coarse-grain it.
        q_micro, p_micro = micro_rk4_step(
            q_micro,
            p_micro,
            DT,
        )

        # Route B:
        # evolve the already-coarse-grained state independently.
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
            BLOCK_SIZE,
        )

        dq, dp, dcombined = state_discrepancy(
            q_micro_coarse,
            p_micro_coarse,
            q_effective,
            p_effective,
        )

        times[step] = step * DT

        q_discrepancy[step] = dq
        p_discrepancy[step] = dp
        combined_discrepancy[step] = dcombined

        micro_coarse_history.append(q_micro_coarse.copy())
        effective_history.append(q_effective.copy())

    micro_coarse_history = np.asarray(micro_coarse_history)
    effective_history = np.asarray(effective_history)

    summary = summarize_discrepancy(
        times,
        combined_discrepancy,
    )

    return {
        "times": times,
        "q_discrepancy": q_discrepancy,
        "p_discrepancy": p_discrepancy,
        "combined_discrepancy": combined_discrepancy,
        "micro_coarse_history": micro_coarse_history,
        "effective_history": effective_history,
        "initial_q_effective": initial_q_effective,
        "initial_p_effective": initial_p_effective,
        "summary": summary,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_outputs(results):
    """Save figures and a text summary to the results directory."""

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    times = results["times"]
    q_discrepancy = results["q_discrepancy"]
    p_discrepancy = results["p_discrepancy"]
    combined_discrepancy = results["combined_discrepancy"]

    micro_history = results["micro_coarse_history"]
    effective_history = results["effective_history"]

    summary = results["summary"]

    # ---------------------------------------------------------------
    # Figure 1: discrepancy
    # ---------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        times,
        q_discrepancy,
        label="q RMS discrepancy",
    )

    plt.plot(
        times,
        p_discrepancy,
        label="p RMS discrepancy",
    )

    plt.plot(
        times,
        combined_discrepancy,
        label="combined discrepancy",
    )

    plt.xlabel("Time")
    plt.ylabel("Discrepancy")
    plt.title("Experiment 001: Coarse-Graining / Evolution Discrepancy")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "experiment_001_delta.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Figure 2: final coarse states
    # ---------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    coarse_x = np.arange(N_COARSE)

    plt.plot(
        coarse_x,
        micro_history[-1],
        label="coarse-grained microscopic state",
    )

    plt.plot(
        coarse_x,
        effective_history[-1],
        "--",
        label="effective state",
    )

    plt.xlabel("Coarse position")
    plt.ylabel("q")
    plt.title("Experiment 001: Final Coarse States")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir / "experiment_001_states.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Text summary
    # ---------------------------------------------------------------

    summary_path = results_dir / "experiment_001_summary.txt"

    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Experiment 001: Micro vs Effective\n")
        f.write("===================================\n\n")

        f.write(f"Microscopic sites: {N_MICRO}\n")
        f.write(f"Block size: {BLOCK_SIZE}\n")
        f.write(f"Coarse sites: {N_COARSE}\n")
        f.write(f"Time step: {DT}\n")
        f.write(f"Final time: {T_FINAL}\n\n")

        f.write("Model parameters\n")
        f.write("----------------\n")
        f.write(f"k = {K}\n")
        f.write(f"omega0^2 = {OMEGA0_SQ}\n")
        f.write(f"beta = {BETA}\n\n")

        f.write("Discrepancy summary\n")
        f.write("-------------------\n")
        f.write(f"Initial: {summary.initial:.10e}\n")
        f.write(f"Final: {summary.final:.10e}\n")
        f.write(f"Maximum: {summary.maximum:.10e}\n")
        f.write(f"Mean: {summary.mean:.10e}\n")
        f.write(f"Time of maximum: {summary.time_of_maximum:.10e}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_experiment()
    save_outputs(results)

    summary = results["summary"]

    print("Experiment 001 complete.")
    print(f"Initial discrepancy: {summary.initial:.6e}")
    print(f"Final discrepancy:   {summary.final:.6e}")
    print(f"Maximum discrepancy: {summary.maximum:.6e}")
    print(f"Time of maximum:     {summary.time_of_maximum:.6e}")
    print("Results written to: results/")
