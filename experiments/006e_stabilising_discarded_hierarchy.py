"""
Experiment 006e: Stabilising Contribution of the Discarded Hierarchy

Question
--------
Can a genuine moment hierarchy contain a higher-order contribution that is
dynamically stabilising, while a natural truncated hierarchy loses that
stabilisation?

Microscopic dynamics
--------------------

    dx/dt = x + x^3 - x^5

The -x^5 term dominates at large |x| and provides stabilisation.

For raw moments

    m_n = <x^n>

the exact hierarchy is

    dm_n/dt = n * (m_n + m_(n+2) - m_(n+4))

In particular,

    dm1/dt = m1 + m3 - m5

    dm3/dt = 3 * (m3 + m5 - m7)

The terms involving m5 and m7 are higher-order information.

This experiment constructs a deliberately truncated hierarchy retaining
m1 and m3 while discarding m5 and above:

    dm1/dt = m1 + m3

    dm3/dt = 3 * m3

This is NOT a fitted closure and does not contain an inserted singular
function.

It is a direct truncation of the exact moment hierarchy.

The experiment asks whether the discarded terms provide an important
stabilising contribution.

Important
---------

The truncated system is intentionally a simple moment truncation. It is
therefore not claimed to be the uniquely correct effective theory.

The purpose is diagnostic:

    complete hierarchy
        |
        | remove higher moments
        v
    truncated hierarchy

If the full system remains bounded while the truncated hierarchy grows
without bound, we have a clean toy example of stabilisation residing in
discarded degrees of freedom.

Outputs
-------

results/006e_summary.txt
results/006e_mean_comparison.png
results/006e_moment_comparison.png
results/006e_stabilising_term.png
results/006e_truncated_phase_space.png
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N = 100_000

DT = 0.001
T = 12.0

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 606


# ---------------------------------------------------------------------------
# Microscopic dynamics
# ---------------------------------------------------------------------------

def microscopic_rhs(x):
    """
    Complete microscopic dynamics.

        dx/dt = x + x^3 - x^5
    """

    return x + x**3 - x**5


def rk4_step(x, dt):
    """
    RK4 step for the complete microscopic ensemble.
    """

    k1 = microscopic_rhs(x)

    k2 = microscopic_rhs(
        x + 0.5 * dt * k1
    )

    k3 = microscopic_rhs(
        x + 0.5 * dt * k2
    )

    k4 = microscopic_rhs(
        x + dt * k3
    )

    return x + (
        dt / 6.0
    ) * (
        k1
        + 2.0 * k2
        + 2.0 * k3
        + k4
    )


# ---------------------------------------------------------------------------
# Moment calculations
# ---------------------------------------------------------------------------

def calculate_moments(x):
    """
    Calculate moments m1 through m7.
    """

    return np.array(
        [
            np.mean(x),
            np.mean(x**2),
            np.mean(x**3),
            np.mean(x**4),
            np.mean(x**5),
            np.mean(x**6),
            np.mean(x**7),
        ]
    )


# ---------------------------------------------------------------------------
# Exact hierarchy diagnostic
# ---------------------------------------------------------------------------

def exact_moment_derivatives(m):
    """
    Calculate the exact hierarchy derivatives for m1 ... m3.

        dm_n/dt = n * (m_n + m_(n+2) - m_(n+4))
    """

    m1, m2, m3, m4, m5, m6, m7 = m

    dm1 = m1 + m3 - m5

    dm2 = 2.0 * (
        m2 + m4 - m6
    )

    dm3 = 3.0 * (
        m3 + m5 - m7
    )

    return dm1, dm2, dm3


# ---------------------------------------------------------------------------
# Truncated hierarchy
# ---------------------------------------------------------------------------

def truncated_rhs(m1, m3):
    """
    Direct truncation of the exact moment hierarchy.

    Exact:

        dm1/dt = m1 + m3 - m5

        dm3/dt = 3 * (m3 + m5 - m7)

    Truncation:

        m5 = 0
        m7 = 0

    therefore:

        dm1/dt = m1 + m3

        dm3/dt = 3 * m3
    """

    dm1 = m1 + m3

    dm3 = 3.0 * m3

    return dm1, dm3


def truncated_rk4_step(m1, m3, dt):
    """
    RK4 integration of the truncated hierarchy.
    """

    k1_m1, k1_m3 = truncated_rhs(
        m1,
        m3,
    )

    k2_m1, k2_m3 = truncated_rhs(
        m1 + 0.5 * dt * k1_m1,
        m3 + 0.5 * dt * k1_m3,
    )

    k3_m1, k3_m3 = truncated_rhs(
        m1 + 0.5 * dt * k2_m1,
        m3 + 0.5 * dt * k2_m3,
    )

    k4_m1, k4_m3 = truncated_rhs(
        m1 + dt * k3_m1,
        m3 + dt * k3_m3,
    )

    new_m1 = m1 + (
        dt / 6.0
    ) * (
        k1_m1
        + 2.0 * k2_m1
        + 2.0 * k3_m1
        + k4_m1
    )

    new_m3 = m3 + (
        dt / 6.0
    ) * (
        k1_m3
        + 2.0 * k2_m3
        + 2.0 * k3_m3
        + k4_m3
    )

    return new_m1, new_m3


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------

def make_initial_ensemble(n, seed=606):
    """
    Initial ensemble centred around x=1.

    The initial state is deliberately moderate, so that the comparison
    begins in a regime where the low-order representation looks reasonable.
    """

    rng = np.random.default_rng(seed)

    x = rng.normal(
        loc=1.0,
        scale=0.12,
        size=n,
    )

    return x


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():

    print("=" * 72)
    print("Experiment 006e: Stabilising Contribution of the Discarded Hierarchy")
    print("=" * 72)

    print()
    print("Microscopic dynamics:")
    print("    dx/dt = x + x^3 - x^5")

    print()
    print("Exact hierarchy:")
    print("    dm_n/dt = n * (m_n + m_(n+2) - m_(n+4))")

    print()
    print("Retained moments:")
    print("    m1, m3")

    print()
    print("Truncated hierarchy:")
    print("    dm1/dt = m1 + m3")
    print("    dm3/dt = 3 * m3")

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    x = make_initial_ensemble(
        N,
        seed=RNG_SEED,
    )

    initial_moments = calculate_moments(x)

    m1_full = initial_moments[0]
    m3_full = initial_moments[2]

    m1_truncated = m1_full
    m3_truncated = m3_full

    print()
    print("Initial moments:")
    print(
        f"    m1 = {initial_moments[0]: .12e}"
    )
    print(
        f"    m2 = {initial_moments[1]: .12e}"
    )
    print(
        f"    m3 = {initial_moments[2]: .12e}"
    )
    print(
        f"    m5 = {initial_moments[4]: .12e}"
    )
    print(
        f"    m7 = {initial_moments[6]: .12e}"
    )

    # ------------------------------------------------------------------
    # Time grid
    # ------------------------------------------------------------------

    steps = int(
        round(T / DT)
    )

    times = (
        np.arange(steps + 1)
        * DT
    )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    m1_history = np.zeros(
        steps + 1
    )

    m3_history = np.zeros(
        steps + 1
    )

    m5_history = np.zeros(
        steps + 1
    )

    m7_history = np.zeros(
        steps + 1
    )

    truncated_m1_history = np.zeros(
        steps + 1
    )

    truncated_m3_history = np.zeros(
        steps + 1
    )

    stabilising_m1_history = np.zeros(
        steps + 1
    )

    stabilising_m3_history = np.zeros(
        steps + 1
    )

    max_abs_x_history = np.zeros(
        steps + 1
    )

    # ------------------------------------------------------------------
    # Initial values
    # ------------------------------------------------------------------

    m1_history[0] = initial_moments[0]
    m3_history[0] = initial_moments[2]
    m5_history[0] = initial_moments[4]
    m7_history[0] = initial_moments[6]

    truncated_m1_history[0] = m1_truncated
    truncated_m3_history[0] = m3_truncated

    stabilising_m1_history[0] = -initial_moments[4]
    stabilising_m3_history[0] = (
        3.0
        * (
            initial_moments[4]
            - initial_moments[6]
        )
    )

    max_abs_x_history[0] = np.max(
        np.abs(x)
    )

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    print()
    print("Evolving...")

    runaway_time = None

    for step in range(1, steps + 1):

        # --------------------------------------------------------------
        # Full microscopic evolution
        # --------------------------------------------------------------

        x = rk4_step(
            x,
            DT,
        )

        current_moments = calculate_moments(x)

        m1 = current_moments[0]
        m3 = current_moments[2]
        m5 = current_moments[4]
        m7 = current_moments[6]

        # --------------------------------------------------------------
        # Truncated hierarchy
        # --------------------------------------------------------------

        (
            m1_truncated,
            m3_truncated,
        ) = truncated_rk4_step(
            m1_truncated,
            m3_truncated,
            DT,
        )

        # --------------------------------------------------------------
        # Store
        # --------------------------------------------------------------

        m1_history[step] = m1
        m3_history[step] = m3
        m5_history[step] = m5
        m7_history[step] = m7

        truncated_m1_history[step] = (
            m1_truncated
        )

        truncated_m3_history[step] = (
            m3_truncated
        )

        # The omitted stabilising contributions are:
        #
        #   -m5
        #
        # in the m1 equation, and
        #
        #   3(m5-m7)
        #
        # in the m3 equation.

        stabilising_m1_history[step] = (
            -m5
        )

        stabilising_m3_history[step] = (
            3.0
            * (
                m5
                - m7
            )
        )

        max_abs_x_history[step] = (
            np.max(
                np.abs(x)
            )
        )

        # --------------------------------------------------------------
        # Detect runaway in truncated representation
        # --------------------------------------------------------------

        if (
            runaway_time is None
            and (
                abs(m1_truncated) > 1e6
                or
                abs(m3_truncated) > 1e6
            )
        ):
            runaway_time = times[step]

        # --------------------------------------------------------------
        # Progress
        # --------------------------------------------------------------

        if step % max(
            1,
            steps // 10,
        ) == 0:

            print(
                f"    t={times[step]:6.3f} "
                f"full m1={m1: .6e} "
                f"truncated m1={m1_truncated: .6e} "
                f"max|x|={max_abs_x_history[step]: .6f}"
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    m1_error = np.abs(
        m1_history
        - truncated_m1_history
    )

    m3_error = np.abs(
        m3_history
        - truncated_m3_history
    )

    maximum_m1_error = np.max(
        m1_error
    )

    final_m1_error = m1_error[-1]

    maximum_m3_error = np.max(
        m3_error
    )

    final_m3_error = m3_error[-1]

    maximum_micro_amplitude = np.max(
        max_abs_x_history
    )

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("Results")
    print("=" * 72)

    print()
    print("Full microscopic system:")
    print(
        f"    maximum |x| = "
        f"{maximum_micro_amplitude:.12e}"
    )

    print()
    print("Truncated hierarchy:")
    print(
        f"    final m1 = "
        f"{m1_truncated:.12e}"
    )

    print(
        f"    final m3 = "
        f"{m3_truncated:.12e}"
    )

    print()
    print("Difference from complete microscopic moments:")
    print(
        f"    maximum m1 error = "
        f"{maximum_m1_error:.12e}"
    )

    print(
        f"    final m1 error = "
        f"{final_m1_error:.12e}"
    )

    print(
        f"    maximum m3 error = "
        f"{maximum_m3_error:.12e}"
    )

    print(
        f"    final m3 error = "
        f"{final_m3_error:.12e}"
    )

    print()
    print("Omitted stabilising contribution:")
    print(
        f"    initial -m5 = "
        f"{stabilising_m1_history[0]:.12e}"
    )

    print(
        f"    maximum |-m5| = "
        f"{np.max(np.abs(stabilising_m1_history)):.12e}"
    )

    print(
        f"    final -m5 = "
        f"{stabilising_m1_history[-1]:.12e}"
    )

    print()

    if runaway_time is None:
        print(
            "The truncated hierarchy did not exceed the "
            "runaway threshold."
        )
    else:
        print(
            f"Truncated hierarchy exceeded |m| > 1e6 at "
            f"t = {runaway_time:.6f}"
        )

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "006e_summary.txt"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Experiment 006e: "
            "Stabilising Contribution of the Discarded Hierarchy\n"
        )

        f.write("=" * 72 + "\n\n")

        f.write(
            "Microscopic dynamics:\n"
        )

        f.write(
            "    dx/dt = x + x^3 - x^5\n\n"
        )

        f.write(
            "Exact hierarchy:\n"
        )

        f.write(
            "    dm_n/dt = n * "
            "(m_n + m_(n+2) - m_(n+4))\n\n"
        )

        f.write(
            "Truncated hierarchy:\n"
        )

        f.write(
            "    dm1/dt = m1 + m3\n"
        )

        f.write(
            "    dm3/dt = 3 * m3\n\n"
        )

        f.write(
            "Initial moments:\n"
        )

        for n, value in enumerate(
            initial_moments,
            start=1,
        ):
            f.write(
                f"    m{n} = "
                f"{value:.12e}\n"
            )

        f.write("\nResults:\n")
        f.write("-" * 72 + "\n")

        f.write(
            f"maximum microscopic |x| = "
            f"{maximum_micro_amplitude:.12e}\n"
        )

        f.write(
            f"maximum m1 error = "
            f"{maximum_m1_error:.12e}\n"
        )

        f.write(
            f"final m1 error = "
            f"{final_m1_error:.12e}\n"
        )

        f.write(
            f"maximum m3 error = "
            f"{maximum_m3_error:.12e}\n"
        )

        f.write(
            f"final m3 error = "
            f"{final_m3_error:.12e}\n"
        )

        f.write(
            f"maximum |-m5| = "
            f"{np.max(np.abs(stabilising_m1_history)):.12e}\n"
        )

        if runaway_time is None:
            f.write(
                "truncated runaway threshold = "
                "not reached\n"
            )
        else:
            f.write(
                f"truncated runaway threshold time = "
                f"{runaway_time:.12e}\n"
            )

        f.write("\nInterpretation:\n")
        f.write("-" * 72 + "\n")

        f.write(
            "The complete microscopic system retains the full hierarchy "
            "of moments.\n"
        )

        f.write(
            "The truncated representation explicitly removes the "
            "higher-order terms m5 and m7.\n"
        )

        f.write(
            "The term -m5 is the higher-order contribution to dm1/dt "
            "associated with the stabilising -x^5 microscopic dynamics.\n"
        )

        f.write(
            "The experiment therefore tests whether removing higher-order "
            "structure removes a dynamically important stabilising effect.\n"
        )

        f.write(
            "This is a hierarchy-truncation experiment, not a claim that "
            "Navier-Stokes itself has this specific structure.\n"
        )

    # ------------------------------------------------------------------
    # Plot 1: mean
    # ------------------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        times,
        m1_history,
        label="Full microscopic m1",
    )

    plt.plot(
        times,
        truncated_m1_history,
        "--",
        label="Truncated hierarchy m1",
    )

    plt.xlabel("Time")
    plt.ylabel("m1")
    plt.title(
        "006e: Complete vs truncated first moment"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "006e_mean_comparison.png",
        dpi=150,
    )

    plt.close()

    # ------------------------------------------------------------------
    # Plot 2: moments
    # ------------------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        times,
        np.abs(m1_history),
        label="|m1|",
    )

    plt.plot(
        times,
        np.abs(m3_history),
        label="|m3|",
    )

    plt.plot(
        times,
        np.abs(m5_history),
        label="|m5|",
    )

    plt.plot(
        times,
        np.abs(m7_history),
        label="|m7|",
    )

    plt.xlabel("Time")
    plt.ylabel("Absolute moment")
    plt.title(
        "006e: Complete moment hierarchy"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "006e_moment_comparison.png",
        dpi=150,
    )

    plt.close()

    # ------------------------------------------------------------------
    # Plot 3: stabilising contribution
    # ------------------------------------------------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        times,
        stabilising_m1_history,
        label="-m5 contribution to dm1/dt",
    )

    plt.plot(
        times,
        stabilising_m3_history,
        label="3(m5-m7) contribution to dm3/dt",
    )

    plt.axhline(
        0.0,
        linewidth=0.8,
    )

    plt.xlabel("Time")
    plt.ylabel("Contribution")
    plt.title(
        "006e: Higher-order contribution removed by truncation"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "006e_stabilising_term.png",
        dpi=150,
    )

    plt.close()

    # ------------------------------------------------------------------
    # Plot 4: truncated phase-space trajectory
    # ------------------------------------------------------------------

    plt.figure(
        figsize=(8, 7)
    )

    plt.plot(
        m1_history,
        m3_history,
        label="Full microscopic moments",
    )

    plt.plot(
        truncated_m1_history,
        truncated_m3_history,
        "--",
        label="Truncated hierarchy",
    )

    plt.xlabel("m1")
    plt.ylabel("m3")
    plt.title(
        "006e: Effective moment-space dynamics"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "006e_truncated_phase_space.png",
        dpi=150,
    )

    plt.close()

    # ------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------

    print()
    print("Saved:")
    print(
        "    results/006e_summary.txt"
    )
    print(
        "    results/006e_mean_comparison.png"
    )
    print(
        "    results/006e_moment_comparison.png"
    )
    print(
        "    results/006e_stabilising_term.png"
    )
    print(
        "    results/006e_truncated_phase_space.png"
    )

    print()
    print("Experiment complete.")


if __name__ == "__main__":
    main()
