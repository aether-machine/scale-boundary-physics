"""
Experiment 002d: Hidden-State Dependence.

This experiment tests whether microscopic information that is invisible
to a coarse-grained representation can nevertheless influence the
future coarse-grained dynamics.

Two microscopic initial states are constructed:

    State A: baseline microscopic state

    State B: same baseline state plus a zero-mean sub-block
              perturbation

The perturbation is constructed so that

    C[X_A(0)] = C[X_B(0)]

even though

    X_A(0) != X_B(0).

Both microscopic states are then evolved independently.

The central quantity is

    S(t) =
        || C[Phi_micro(t, X_A)]
         - C[Phi_micro(t, X_B)] ||

If S(t) becomes nonzero, then microscopic information that is absent
from the coarse representation influences the subsequent coarse
dynamics.

The experiment also compares both microscopic trajectories with the
same effective trajectory.

This is a controlled toy-model experiment. It does not establish
anything about physical fluids by itself.
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

BLOCK_SIZES = [
    2,
    4,
    8,
    16,
]

HIDDEN_AMPLITUDES = [
    0.05,
    0.10,
    0.20,
    0.40,
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
        k1_q
        + 2.0 * k2_q
        + 2.0 * k3_q
        + k4_q
    )

    p_next = p + (dt / 6.0) * (
        k1_p
        + 2.0 * k2_p
        + 2.0 * k3_p
        + k4_p
    )

    return q_next, p_next


# ---------------------------------------------------------------------------
# Initial conditions
# ---------------------------------------------------------------------------

def baseline_initial_condition(n):
    """
    Construct the resolved baseline state.

    This is identical for State A and State B before the hidden
    perturbation is added.
    """

    x = np.arange(n)

    q0 = AMPLITUDE * np.sin(
        2.0 * np.pi * WAVELENGTHS * x / n
    )

    p0 = np.zeros_like(q0)

    return q0, p0


def hidden_perturbation(n, block_size, amplitude):
    """
    Construct a perturbation whose block average is exactly zero.

    Within each coarse block the pattern is

        [+A, -A, +A, -A, ...]

    for even block sizes.

    Therefore:

        C[delta_q] = 0

    exactly, apart from floating-point roundoff.

    Momentum is unchanged.
    """

    if block_size % 2 != 0:
        raise ValueError(
            "Block size must be even for the zero-mean "
            "alternating perturbation."
        )

    delta_q = np.zeros(n)

    pattern = np.array(
        [
            amplitude if i % 2 == 0 else -amplitude
            for i in range(block_size)
        ]
    )

    for start in range(0, n, block_size):

        end = start + block_size

        delta_q[start:end] = pattern

    delta_p = np.zeros(n)

    return delta_q, delta_p


def construct_initial_states(
    block_size,
    hidden_amplitude,
):
    """
    Construct States A and B.

    State A:
        baseline

    State B:
        baseline + hidden zero-mean perturbation
    """

    q_a, p_a = baseline_initial_condition(
        N_MICRO
    )

    delta_q, delta_p = hidden_perturbation(
        N_MICRO,
        block_size,
        hidden_amplitude,
    )

    q_b = q_a + delta_q
    p_b = p_a + delta_p

    return q_a, p_a, q_b, p_b


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_single_case(
    block_size,
    hidden_amplitude,
):
    """
    Run one hidden-state experiment.

    Returns time series for:

        hidden-state separation
        closure error A
        closure error B
    """

    q_a, p_a, q_b, p_b = construct_initial_states(
        block_size,
        hidden_amplitude,
    )

    # Coarse representations at t=0.
    q_coarse_a, p_coarse_a = coarse_grain(
        q_a,
        p_a,
        block_size,
    )

    q_coarse_b, p_coarse_b = coarse_grain(
        q_b,
        p_b,
        block_size,
    )

    # Verify the defining condition of the experiment.
    initial_difference = state_discrepancy(
        q_coarse_a,
        p_coarse_a,
        q_coarse_b,
        p_coarse_b,
    )[2]

    if initial_difference > 1e-12:
        raise RuntimeError(
            "Initial coarse states are not identical. "
            f"Difference = {initial_difference}"
        )

    # Effective model begins from the common coarse state.
    q_effective = q_coarse_a.copy()
    p_effective = p_coarse_a.copy()

    n_steps = int(round(T_FINAL / DT))

    times = []

    hidden_state_separation = []
    closure_error_a = []
    closure_error_b = []

    for step in range(n_steps + 1):

        t = step * DT

        # Coarse representations of the two microscopic states.
        q_a_coarse, p_a_coarse = coarse_grain(
            q_a,
            p_a,
            block_size,
        )

        q_b_coarse, p_b_coarse = coarse_grain(
            q_b,
            p_b,
            block_size,
        )

        # Hidden-state separation.
        _, _, separation = state_discrepancy(
            q_a_coarse,
            p_a_coarse,
            q_b_coarse,
            p_b_coarse,
        )

        # Closure error of State A.
        _, _, error_a = state_discrepancy(
            q_a_coarse,
            p_a_coarse,
            q_effective,
            p_effective,
        )

        # Closure error of State B.
        _, _, error_b = state_discrepancy(
            q_b_coarse,
            p_b_coarse,
            q_effective,
            p_effective,
        )

        times.append(t)

        hidden_state_separation.append(
            separation
        )

        closure_error_a.append(error_a)
        closure_error_b.append(error_b)

        if step == n_steps:
            break

        # Advance microscopic State A.
        q_a, p_a = micro_rk4_step(
            q_a,
            p_a,
            DT,
        )

        # Advance microscopic State B.
        q_b, p_b = micro_rk4_step(
            q_b,
            p_b,
            DT,
        )

        # Advance the single effective state.
        q_effective, p_effective = effective_rk4_step(
            q_effective,
            p_effective,
            DT,
            k=K,
            omega0_sq=OMEGA0_SQ,
            beta=BETA,
        )

    return {
        "times": np.asarray(times),
        "hidden_state_separation":
            np.asarray(hidden_state_separation),
        "closure_error_a":
            np.asarray(closure_error_a),
        "closure_error_b":
            np.asarray(closure_error_b),
        "initial_difference":
            initial_difference,
    }


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def run_experiment():
    """Run all block-size and hidden-amplitude combinations."""

    results = []

    for block_size in BLOCK_SIZES:

        for hidden_amplitude in HIDDEN_AMPLITUDES:

            print(
                f"Running block={block_size:2d}, "
                f"hidden_amplitude="
                f"{hidden_amplitude:.2f}"
            )

            data = run_single_case(
                block_size,
                hidden_amplitude,
            )

            separation = data[
                "hidden_state_separation"
            ]

            error_a = data[
                "closure_error_a"
            ]

            error_b = data[
                "closure_error_b"
            ]

            maximum_separation = np.max(
                separation
            )

            final_separation = separation[-1]

            maximum_error_a = np.max(error_a)
            maximum_error_b = np.max(error_b)

            results.append(
                {
                    "block_size": block_size,
                    "hidden_amplitude":
                        hidden_amplitude,
                    "times": data["times"],
                    "hidden_state_separation":
                        separation,
                    "closure_error_a":
                        error_a,
                    "closure_error_b":
                        error_b,
                    "initial_difference":
                        data["initial_difference"],
                    "maximum_separation":
                        maximum_separation,
                    "final_separation":
                        final_separation,
                    "maximum_error_a":
                        maximum_error_a,
                    "maximum_error_b":
                        maximum_error_b,
                }
            )

            print(
                f"    initial coarse difference = "
                f"{data['initial_difference']:.6e}"
            )

            print(
                f"    maximum hidden-state separation = "
                f"{maximum_separation:.6e}"
            )

            print(
                f"    final hidden-state separation = "
                f"{final_separation:.6e}"
            )

            print(
                f"    maximum closure error A = "
                f"{maximum_error_a:.6e}"
            )

            print(
                f"    maximum closure error B = "
                f"{maximum_error_b:.6e}"
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
        / "experiment_002d_results.csv"
    )

    with csv_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "block_size,"
            "hidden_amplitude,"
            "initial_difference,"
            "maximum_separation,"
            "final_separation,"
            "maximum_error_a,"
            "maximum_error_b\n"
        )

        for result in results:

            f.write(
                f"{result['block_size']},"
                f"{result['hidden_amplitude']:.6f},"
                f"{result['initial_difference']:.10e},"
                f"{result['maximum_separation']:.10e},"
                f"{result['final_separation']:.10e},"
                f"{result['maximum_error_a']:.10e},"
                f"{result['maximum_error_b']:.10e}\n"
            )

    # ---------------------------------------------------------------
    # Plot 1: hidden-state separation
    # ---------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for result in results:

        label = (
            f"B={result['block_size']}, "
            f"A={result['hidden_amplitude']:.2f}"
        )

        plt.plot(
            result["times"],
            result["hidden_state_separation"],
            label=label,
        )

    plt.xlabel("Time")
    plt.ylabel("Coarse-state separation")
    plt.title(
        "Experiment 002d: Hidden-State Separation"
    )
    plt.legend(
        ncol=2,
        fontsize=8,
    )
    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_002d_hidden_state.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Plot 2: closure errors
    # ---------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for result in results:

        label_a = (
            f"A, B={result['block_size']}, "
            f"amp={result['hidden_amplitude']:.2f}"
        )

        label_b = (
            f"B, B={result['block_size']}, "
            f"amp={result['hidden_amplitude']:.2f}"
        )

        plt.plot(
            result["times"],
            result["closure_error_a"],
            label=label_a,
        )

        plt.plot(
            result["times"],
            result["closure_error_b"],
            linestyle="--",
            label=label_b,
        )

    plt.xlabel("Time")
    plt.ylabel("Closure error")
    plt.title(
        "Experiment 002d: Effective-Model Closure Error"
    )
    plt.legend(
        ncol=2,
        fontsize=7,
    )
    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_002d_closure_error.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Plot 3: maximum separation versus amplitude
    # ---------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for block_size in BLOCK_SIZES:

        amplitudes = []
        maximum_separations = []

        for result in results:

            if result["block_size"] == block_size:

                amplitudes.append(
                    result["hidden_amplitude"]
                )

                maximum_separations.append(
                    result["maximum_separation"]
                )

        plt.plot(
            amplitudes,
            maximum_separations,
            marker="o",
            label=f"Block size {block_size}",
        )

    plt.xlabel("Hidden perturbation amplitude")
    plt.ylabel("Maximum coarse-state separation")
    plt.title(
        "Experiment 002d: Hidden-State Influence"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        results_dir
        / "experiment_002d_amplitude.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------------
    # Text summary
    # ---------------------------------------------------------------

    summary_path = (
        results_dir
        / "experiment_002d_summary.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Experiment 002d: Hidden-State Dependence\n"
        )

        f.write(
            "==========================================\n\n"
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
            "The defining condition is:\n\n"
        )

        f.write(
            "C[X_A(0)] = C[X_B(0)]\n\n"
        )

        f.write(
            "while:\n\n"
        )

        f.write(
            "X_A(0) != X_B(0)\n\n"
        )

        f.write(
            "Results\n"
            "-------\n"
        )

        for result in results:

            f.write(
                f"block={result['block_size']:2d}, "
                f"hidden_amplitude="
                f"{result['hidden_amplitude']:.2f}, "
                f"initial_difference="
                f"{result['initial_difference']:.10e}, "
                f"maximum_separation="
                f"{result['maximum_separation']:.10e}, "
                f"final_separation="
                f"{result['final_separation']:.10e}, "
                f"maximum_error_A="
                f"{result['maximum_error_a']:.10e}, "
                f"maximum_error_B="
                f"{result['maximum_error_b']:.10e}\n"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    results = run_experiment()

    save_outputs(results)

    print()
    print("Experiment 002d complete.")
    print("Results written to: results/")
