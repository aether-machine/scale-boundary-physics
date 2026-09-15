"""
Experiment 006g: Intrinsic Non-Closure

Question:
Can two microscopic states have exactly the same retained variables
but subsequently produce different trajectories in those variables?

If so, an exact autonomous effective equation

    dH/dt = F(H)

cannot exist using only the retained variables.

This is a stronger test than closure quality.

The experiment uses the microscopic system

    dx/dt = x + x^3 - x^5

and compares two ensembles with identical low-order moments

    m1 = E[x]
    m2 = E[x^2]

but different higher-order structure.

Both ensembles are then evolved using the same microscopic dynamics.

We measure:

    1. initial low-order agreement
    2. hidden higher-moment difference
    3. subsequent m1/m2 separation
    4. divergence of the exact low-order derivatives

The key diagnostic is:

    same retained state
        +
    different future
        =
    non-closure

Outputs:

    results/006g_intrinsic_nonclosure.csv
    results/006g_intrinsic_nonclosure_summary.txt
    results/006g_low_order_separation.png
    results/006g_hidden_moment_difference.png
    results/006g_derivative_difference.png
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

N = 200_000

DT = 0.002
T = 8.0

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = OUTPUT_DIR / "006g_intrinsic_nonclosure.csv"
SUMMARY_PATH = OUTPUT_DIR / "006g_intrinsic_nonclosure_summary.txt"

PLOT_LOW_ORDER = OUTPUT_DIR / "006g_low_order_separation.png"
PLOT_HIDDEN = OUTPUT_DIR / "006g_hidden_moment_difference.png"
PLOT_DERIVATIVE = OUTPUT_DIR / "006g_derivative_difference.png"


# ---------------------------------------------------------------------
# Microscopic dynamics
# ---------------------------------------------------------------------

def microscopic_rhs(x):
    """
    dx/dt = x + x^3 - x^5
    """
    return x + x**3 - x**5


def rk4_step(x, dt):
    """
    RK4 step for the microscopic ensemble.
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


# ---------------------------------------------------------------------
# Moment calculation
# ---------------------------------------------------------------------

def moments(x, max_order=8):
    """
    Return raw moments m0 ... m_max_order.
    """
    return np.array(
        [
            np.mean(x ** n)
            for n in range(max_order + 1)
        ],
        dtype=float,
    )


def moment_derivative(m, n):
    """
    Exact moment hierarchy:

        dm_n/dt = n * (m_n + m_(n+2) - m_(n+4))
    """
    return n * (
        m[n]
        + m[n + 2]
        - m[n + 4]
    )


# ---------------------------------------------------------------------
# Construct two ensembles with identical m1 and m2
# ---------------------------------------------------------------------

def make_ensembles():
    """
    Construct two ensembles with exactly the same mean and variance,
    but deliberately different higher-order structure.

    Ensemble A:
        approximately Gaussian

    Ensemble B:
        symmetric three-component mixture

    Both are subsequently affine-normalised so that their sample
    mean and variance agree exactly.

    This means the retained variables are identical to numerical
    precision, while the higher moments differ.
    """

    rng = np.random.default_rng(12345)

    # ---------------------------------------------------------------
    # Ensemble A
    # ---------------------------------------------------------------

    x_a = rng.normal(
        loc=1.0,
        scale=0.20,
        size=N,
    )

    # ---------------------------------------------------------------
    # Ensemble B
    # ---------------------------------------------------------------

    n_each = N // 3

    b1 = rng.normal(
        loc=0.70,
        scale=0.08,
        size=n_each,
    )

    b2 = rng.normal(
        loc=1.00,
        scale=0.08,
        size=n_each,
    )

    b3 = rng.normal(
        loc=1.30,
        scale=0.08,
        size=N - 2 * n_each,
    )

    x_b = np.concatenate(
        [b1, b2, b3]
    )

    # ---------------------------------------------------------------
    # Force exact equality of first two moments
    # ---------------------------------------------------------------

    target_mean = np.mean(x_a)
    target_variance = np.var(
        x_a,
        ddof=0,
    )

    # Normalise B
    b_mean = np.mean(x_b)
    b_std = np.std(x_b)

    x_b = (
        (x_b - b_mean)
        / b_std
    )

    x_b = (
        x_b * np.sqrt(target_variance)
        + target_mean
    )

    # Finally remove tiny floating-point differences by applying
    # a second affine correction.
    #
    # This gives equality at machine precision without altering the
    # qualitative higher-order structure.

    b_mean = np.mean(x_b)
    b_variance = np.var(
        x_b,
        ddof=0,
    )

    x_b = (
        (x_b - b_mean)
        * np.sqrt(
            target_variance
            / b_variance
        )
        + target_mean
    )

    return x_a, x_b


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():

    print(
        "Experiment 006g: Intrinsic Non-Closure"
    )
    print()

    print(
        "Microscopic dynamics:"
    )
    print(
        "    dx/dt = x + x^3 - x^5"
    )
    print()

    print(
        "Retained variables:"
    )
    print(
        "    m1 = E[x]"
    )
    print(
        "    m2 = E[x^2]"
    )
    print()

    print(
        "Question:"
    )
    print(
        "Can identical retained states have different futures?"
    )
    print()

    # -----------------------------------------------------------------
    # Initial ensembles
    # -----------------------------------------------------------------

    x_a, x_b = make_ensembles()

    m_a = moments(x_a)
    m_b = moments(x_b)

    print("Initial states")
    print(
        f"m1 A = {m_a[1]:.15e}"
    )
    print(
        f"m1 B = {m_b[1]:.15e}"
    )

    print(
        f"m2 A = {m_a[2]:.15e}"
    )
    print(
        f"m2 B = {m_b[2]:.15e}"
    )

    print()

    print(
        "Initial retained-state differences:"
    )

    print(
        f"|m1 A - m1 B| = "
        f"{abs(m_a[1] - m_b[1]):.15e}"
    )

    print(
        f"|m2 A - m2 B| = "
        f"{abs(m_a[2] - m_b[2]):.15e}"
    )

    print()

    print(
        "Initial hidden-state differences:"
    )

    for n in [3, 4, 5, 6, 7, 8]:

        print(
            f"|m{n} A - m{n} B| = "
            f"{abs(m_a[n] - m_b[n]):.15e}"
        )

    print()

    # -----------------------------------------------------------------
    # Time series
    # -----------------------------------------------------------------

    times = []

    m1_a_values = []
    m1_b_values = []

    m2_a_values = []
    m2_b_values = []

    m3_a_values = []
    m3_b_values = []

    m5_a_values = []
    m5_b_values = []

    derivative_m1_a = []
    derivative_m1_b = []

    derivative_m2_a = []
    derivative_m2_b = []

    retained_separation = []
    hidden_separation = []
    derivative_separation = []

    max_abs_x_a = []
    max_abs_x_b = []

    steps = int(
        T / DT
    )

    for step in range(steps + 1):

        t = step * DT

        m_a = moments(
            x_a,
            max_order=8,
        )

        m_b = moments(
            x_b,
            max_order=8,
        )

        # -------------------------------------------------------------
        # Exact derivatives implied by the full hierarchy
        # -------------------------------------------------------------

        dm1_a = moment_derivative(
            m_a,
            1,
        )

        dm1_b = moment_derivative(
            m_b,
            1,
        )

        dm2_a = moment_derivative(
            m_a,
            2,
        )

        dm2_b = moment_derivative(
            m_b,
            2,
        )

        # -------------------------------------------------------------
        # Diagnostics
        # -------------------------------------------------------------

        retained_diff = np.sqrt(
            (m_a[1] - m_b[1]) ** 2
            +
            (m_a[2] - m_b[2]) ** 2
        )

        hidden_diff = np.sqrt(
            (m_a[3] - m_b[3]) ** 2
            +
            (m_a[4] - m_b[4]) ** 2
            +
            (m_a[5] - m_b[5]) ** 2
        )

        derivative_diff = np.sqrt(
            (dm1_a - dm1_b) ** 2
            +
            (dm2_a - dm2_b) ** 2
        )

        times.append(t)

        m1_a_values.append(m_a[1])
        m1_b_values.append(m_b[1])

        m2_a_values.append(m_a[2])
        m2_b_values.append(m_b[2])

        m3_a_values.append(m_a[3])
        m3_b_values.append(m_b[3])

        m5_a_values.append(m_a[5])
        m5_b_values.append(m_b[5])

        derivative_m1_a.append(dm1_a)
        derivative_m1_b.append(dm1_b)

        derivative_m2_a.append(dm2_a)
        derivative_m2_b.append(dm2_b)

        retained_separation.append(
            retained_diff
        )

        hidden_separation.append(
            hidden_diff
        )

        derivative_separation.append(
            derivative_diff
        )

        max_abs_x_a.append(
            np.max(np.abs(x_a))
        )

        max_abs_x_b.append(
            np.max(np.abs(x_b))
        )

        if step == steps:
            break

        # -------------------------------------------------------------
        # Evolve both microscopic systems
        # -------------------------------------------------------------

        x_a = rk4_step(
            x_a,
            DT,
        )

        x_b = rk4_step(
            x_b,
            DT,
        )

        if (
            not np.all(
                np.isfinite(x_a)
            )
            or
            not np.all(
                np.isfinite(x_b)
            )
        ):
            print(
                "Microscopic integration "
                "became unstable."
            )
            break

    # -----------------------------------------------------------------
    # Convert to arrays
    # -----------------------------------------------------------------

    times = np.asarray(times)

    m1_a_values = np.asarray(
        m1_a_values
    )
    m1_b_values = np.asarray(
        m1_b_values
    )

    m2_a_values = np.asarray(
        m2_a_values
    )
    m2_b_values = np.asarray(
        m2_b_values
    )

    m3_a_values = np.asarray(
        m3_a_values
    )
    m3_b_values = np.asarray(
        m3_b_values
    )

    m5_a_values = np.asarray(
        m5_a_values
    )
    m5_b_values = np.asarray(
        m5_b_values
    )

    derivative_m1_a = np.asarray(
        derivative_m1_a
    )
    derivative_m1_b = np.asarray(
        derivative_m1_b
    )

    derivative_m2_a = np.asarray(
        derivative_m2_a
    )
    derivative_m2_b = np.asarray(
        derivative_m2_b
    )

    retained_separation = np.asarray(
        retained_separation
    )

    hidden_separation = np.asarray(
        hidden_separation
    )

    derivative_separation = np.asarray(
        derivative_separation
    )

    max_abs_x_a = np.asarray(
        max_abs_x_a
    )

    max_abs_x_b = np.asarray(
        max_abs_x_b
    )

    # -----------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------

    initial_retained_difference = (
        retained_separation[0]
    )

    maximum_retained_difference = (
        np.max(retained_separation)
    )

    final_retained_difference = (
        retained_separation[-1]
    )

    maximum_hidden_difference = (
        np.max(hidden_separation)
    )

    final_hidden_difference = (
        hidden_separation[-1]
    )

    maximum_derivative_difference = (
        np.max(derivative_separation)
    )

    final_derivative_difference = (
        derivative_separation[-1]
    )

    maximum_micro_a = (
        np.max(max_abs_x_a)
    )

    maximum_micro_b = (
        np.max(max_abs_x_b)
    )

    # -----------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------

    data = np.column_stack(
        [
            times,

            m1_a_values,
            m1_b_values,

            m2_a_values,
            m2_b_values,

            m3_a_values,
            m3_b_values,

            m5_a_values,
            m5_b_values,

            derivative_m1_a,
            derivative_m1_b,

            derivative_m2_a,
            derivative_m2_b,

            retained_separation,
            hidden_separation,
            derivative_separation,

            max_abs_x_a,
            max_abs_x_b,
        ]
    )

    header = (
        "time,"
        "m1_A,m1_B,"
        "m2_A,m2_B,"
        "m3_A,m3_B,"
        "m5_A,m5_B,"
        "dm1_A,dm1_B,"
        "dm2_A,dm2_B,"
        "retained_separation,"
        "hidden_separation,"
        "derivative_separation,"
        "max_abs_x_A,"
        "max_abs_x_B"
    )

    np.savetxt(
        CSV_PATH,
        data,
        delimiter=",",
        header=header,
        comments="",
    )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Experiment 006g: Intrinsic Non-Closure\n\n"
        )

        f.write(
            "Microscopic dynamics:\n"
            "    dx/dt = x + x^3 - x^5\n\n"
        )

        f.write(
            "Retained variables:\n"
            "    m1 = E[x]\n"
            "    m2 = E[x^2]\n\n"
        )

        f.write(
            "Initial retained-state differences:\n"
            f"    m1 difference = "
            f"{abs(m_a[1] - m_b[1]):.12e}\n"
            f"    m2 difference = "
            f"{abs(m_a[2] - m_b[2]):.12e}\n\n"
        )

        f.write(
            "Results:\n"
            f"initial retained separation = "
            f"{initial_retained_difference:.12e}\n"
            f"maximum retained separation = "
            f"{maximum_retained_difference:.12e}\n"
            f"final retained separation = "
            f"{final_retained_difference:.12e}\n"
            f"maximum hidden separation = "
            f"{maximum_hidden_difference:.12e}\n"
            f"final hidden separation = "
            f"{final_hidden_difference:.12e}\n"
            f"maximum derivative separation = "
            f"{maximum_derivative_difference:.12e}\n"
            f"final derivative separation = "
            f"{final_derivative_difference:.12e}\n"
            f"maximum microscopic |x| A = "
            f"{maximum_micro_a:.12e}\n"
            f"maximum microscopic |x| B = "
            f"{maximum_micro_b:.12e}\n"
        )

        f.write("\nInterpretation:\n")

        f.write(
            "If the retained variables begin identically but "
            "subsequently diverge, then the retained variables "
            "do not define an exact autonomous dynamical state.\n"
        )

        f.write(
            "The difference arises because the exact evolution "
            "of the retained moments depends on higher moments "
            "that were discarded from the representation.\n"
        )

    # -----------------------------------------------------------------
    # Plot 1: retained variables
    # -----------------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        times,
        m1_a_values,
        label="A: m1",
    )

    plt.plot(
        times,
        m1_b_values,
        "--",
        label="B: m1",
    )

    plt.plot(
        times,
        m2_a_values,
        label="A: m2",
    )

    plt.plot(
        times,
        m2_b_values,
        "--",
        label="B: m2",
    )

    plt.xlabel("Time")
    plt.ylabel("Moment")
    plt.title(
        "006g: Initially Identical Retained Variables"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        PLOT_LOW_ORDER,
        dpi=160,
    )
    plt.close()

    # -----------------------------------------------------------------
    # Plot 2: hidden moments
    # -----------------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        times,
        np.abs(
            m3_a_values
            - m3_b_values
        ),
        label="|m3 A - m3 B|",
    )

    plt.plot(
        times,
        np.abs(
            m5_a_values
            - m5_b_values
        ),
        label="|m5 A - m5 B|",
    )

    plt.xlabel("Time")
    plt.ylabel("Hidden moment difference")
    plt.title(
        "006g: Hidden-State Separation"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        PLOT_HIDDEN,
        dpi=160,
    )
    plt.close()

    # -----------------------------------------------------------------
    # Plot 3: derivative difference
    # -----------------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        times,
        derivative_separation,
        label="Difference in exact retained derivatives",
    )

    plt.xlabel("Time")
    plt.ylabel("Derivative separation")
    plt.title(
        "006g: Different Futures from the Same Retained State"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        PLOT_DERIVATIVE,
        dpi=160,
    )
    plt.close()

    # -----------------------------------------------------------------
    # Console output
    # -----------------------------------------------------------------

    print()
    print("Results:")

    print(
        f"initial retained separation = "
        f"{initial_retained_difference:.12e}"
    )

    print(
        f"maximum retained separation = "
        f"{maximum_retained_difference:.12e}"
    )

    print(
        f"final retained separation = "
        f"{final_retained_difference:.12e}"
    )

    print(
        f"maximum hidden separation = "
        f"{maximum_hidden_difference:.12e}"
    )

    print(
        f"final hidden separation = "
        f"{final_hidden_difference:.12e}"
    )

    print(
        f"maximum derivative separation = "
        f"{maximum_derivative_difference:.12e}"
    )

    print(
        f"final derivative separation = "
        f"{final_derivative_difference:.12e}"
    )

    print(
        f"maximum microscopic |x| A = "
        f"{maximum_micro_a:.12e}"
    )

    print(
        f"maximum microscopic |x| B = "
        f"{maximum_micro_b:.12e}"
    )

    print()
    print("Outputs:")
    print(f"    {CSV_PATH}")
    print(f"    {SUMMARY_PATH}")
    print(f"    {PLOT_LOW_ORDER}")
    print(f"    {PLOT_HIDDEN}")
    print(f"    {PLOT_DERIVATIVE}")


if __name__ == "__main__":
    main()
