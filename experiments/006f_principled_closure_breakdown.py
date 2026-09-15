"""
Experiment 006f: Principled Closure Breakdown

Question:
Can a natural moment closure remain accurate while discarded
higher-order information is dynamically weak, and then lose accuracy
when that information becomes important?

Microscopic dynamics:
    dx/dt = x + x^3 - x^5

Exact moment hierarchy:
    dm_n/dt = n * (m_n + m_(n+2) - m_(n+4))

We retain only m1 and m2 and construct a Gaussian closure
for higher moments.

For a Gaussian random variable with mean mu and variance sigma^2,
higher raw moments are determined by mu and sigma^2.

The experiment compares:
    1. Full microscopic ensemble evolution
    2. Exact low-order moments of that ensemble
    3. Gaussian-closed effective dynamics

The closure is NOT fitted to the trajectory and does not contain
a deliberately inserted singularity.

The main diagnostic is whether the Gaussian estimate of the
discarded stabilising moment m5 remains accurate as the dynamics
evolve.

Outputs:
    results/006f_closure_breakdown.csv
    results/006f_closure_breakdown_summary.txt
    results/006f_closure_error.png
    results/006f_stabilising_moment.png
    results/006f_moment_comparison.png
"""

from pathlib import Path
import math

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

CSV_PATH = OUTPUT_DIR / "006f_closure_breakdown.csv"
SUMMARY_PATH = OUTPUT_DIR / "006f_closure_breakdown_summary.txt"

PLOT_ERROR = OUTPUT_DIR / "006f_closure_error.png"
PLOT_STABILISING = OUTPUT_DIR / "006f_stabilising_moment.png"
PLOT_MOMENTS = OUTPUT_DIR / "006f_moment_comparison.png"


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
    One RK4 step applied independently to every microscopic state.
    """
    k1 = microscopic_rhs(x)
    k2 = microscopic_rhs(x + 0.5 * dt * k1)
    k3 = microscopic_rhs(x + 0.5 * dt * k2)
    k4 = microscopic_rhs(x + dt * k3)

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ---------------------------------------------------------------------
# Gaussian raw moments
# ---------------------------------------------------------------------

def gaussian_raw_moment(n, mu, variance):
    """
    Return the nth raw moment of a Gaussian distribution.

    E[X^n] =
        sum over k:
            n! / ((n-2k)! k! 2^k)
            * mu^(n-2k)
            * variance^k
    """
    variance = max(float(variance), 0.0)

    total = 0.0

    for k in range(n // 2 + 1):
        coefficient = (
            math.factorial(n)
            / (
                math.factorial(n - 2 * k)
                * math.factorial(k)
                * (2.0 ** k)
            )
        )

        total += (
            coefficient
            * mu ** (n - 2 * k)
            * variance ** k
        )

    return total


def gaussian_moments(mu, variance, max_order=8):
    """
    Return Gaussian raw moments m0 ... m_max_order.
    """
    return np.array(
        [
            gaussian_raw_moment(n, mu, variance)
            for n in range(max_order + 1)
        ],
        dtype=float,
    )


# ---------------------------------------------------------------------
# Exact moment derivatives
# ---------------------------------------------------------------------

def exact_moment_derivative(moments, n):
    """
    Exact hierarchy:

        dm_n/dt = n * (m_n + m_(n+2) - m_(n+4))
    """
    return n * (
        moments[n]
        + moments[n + 2]
        - moments[n + 4]
    )


# ---------------------------------------------------------------------
# Gaussian-closed effective dynamics
# ---------------------------------------------------------------------

def gaussian_closed_rhs(m1, m2):
    """
    Closed dynamics for m1 and m2.

    The retained variables are:
        m1 = E[x]
        m2 = E[x^2]

    From these:
        variance = m2 - m1^2

    Higher moments are supplied by the Gaussian closure.
    """

    variance = max(m2 - m1**2, 0.0)

    gm = gaussian_moments(
        mu=m1,
        variance=variance,
        max_order=6,
    )

    dm1 = exact_moment_derivative(gm, 1)
    dm2 = exact_moment_derivative(gm, 2)

    return dm1, dm2, gm


def gaussian_closed_rk4_step(m1, m2, dt):
    """
    RK4 integration of the two-dimensional Gaussian-closed system.
    """

    def rhs(a, b):
        dm1, dm2, _ = gaussian_closed_rhs(a, b)
        return dm1, dm2

    k1_1, k1_2 = rhs(m1, m2)

    k2_1, k2_2 = rhs(
        m1 + 0.5 * dt * k1_1,
        m2 + 0.5 * dt * k1_2,
    )

    k3_1, k3_2 = rhs(
        m1 + 0.5 * dt * k2_1,
        m2 + 0.5 * dt * k2_2,
    )

    k4_1, k4_2 = rhs(
        m1 + dt * k3_1,
        m2 + dt * k3_2,
    )

    new_m1 = m1 + (dt / 6.0) * (
        k1_1 + 2*k2_1 + 2*k3_1 + k4_1
    )

    new_m2 = m2 + (dt / 6.0) * (
        k1_2 + 2*k2_2 + 2*k3_2 + k4_2
    )

    return new_m1, new_m2


# ---------------------------------------------------------------------
# Initial microscopic ensemble
# ---------------------------------------------------------------------

def make_initial_ensemble():
    """
    Construct a deliberately non-Gaussian but well-behaved ensemble.

    The distribution is a mixture of two narrow Gaussian populations.
    This gives us a controlled way to test the Gaussian closure without
    fitting the closure to the subsequent trajectory.
    """

    rng = np.random.default_rng(12345)

    n1 = N // 2
    n2 = N - n1

    x1 = rng.normal(
        loc=0.90,
        scale=0.12,
        size=n1,
    )

    x2 = rng.normal(
        loc=1.10,
        scale=0.12,
        size=n2,
    )

    x = np.concatenate([x1, x2])

    return x


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():

    print("Experiment 006f: Principled Closure Breakdown")
    print()
    print("Microscopic dynamics:")
    print("    dx/dt = x + x^3 - x^5")
    print()
    print("Exact moment hierarchy:")
    print("    dm_n/dt = n * (m_n + m_(n+2) - m_(n+4))")
    print()
    print("Effective closure:")
    print("    Gaussian closure using only m1 and m2")
    print()

    x = make_initial_ensemble()

    initial_m1 = np.mean(x)
    initial_m2 = np.mean(x**2)

    print("Initial ensemble:")
    print(f"    m1 = {initial_m1:.12e}")
    print(f"    m2 = {initial_m2:.12e}")
    print(
        f"    variance = "
        f"{initial_m2 - initial_m1**2:.12e}"
    )
    print()

    # Effective variables
    effective_m1 = initial_m1
    effective_m2 = initial_m2

    times = []
    exact_m1_values = []
    exact_m2_values = []
    exact_m5_values = []

    closure_m1_values = []
    closure_m2_values = []
    closure_m5_values = []

    closure_errors = []
    stabilising_exact = []
    stabilising_closure = []

    max_abs_x = []
    failure_time = None

    steps = int(T / DT)

    for step in range(steps + 1):

        t = step * DT

        # Exact microscopic moments
        exact_m1 = np.mean(x)
        exact_m2 = np.mean(x**2)
        exact_m5 = np.mean(x**5)

        # Gaussian reconstruction from effective variables
        variance = max(
            effective_m2 - effective_m1**2,
            0.0,
        )

        gm = gaussian_moments(
            effective_m1,
            variance,
            max_order=6,
        )

        closure_m5 = gm[5]

        # Diagnostics
        error = np.sqrt(
            (exact_m1 - effective_m1)**2
            + (exact_m2 - effective_m2)**2
        )

        times.append(t)

        exact_m1_values.append(exact_m1)
        exact_m2_values.append(exact_m2)
        exact_m5_values.append(exact_m5)

        closure_m1_values.append(effective_m1)
        closure_m2_values.append(effective_m2)
        closure_m5_values.append(closure_m5)

        closure_errors.append(error)

        # The stabilising contribution in dm1/dt is -m5.
        stabilising_exact.append(-exact_m5)
        stabilising_closure.append(-closure_m5)

        max_abs_x.append(np.max(np.abs(x)))

        if failure_time is None:
            if (
                not np.isfinite(effective_m1)
                or not np.isfinite(effective_m2)
                or abs(effective_m1) > 1e6
                or abs(effective_m2) > 1e12
            ):
                failure_time = t

        if step == steps:
            break

        # Advance microscopic system
        x = rk4_step(x, DT)

        # Advance Gaussian closure
        effective_m1, effective_m2 = (
            gaussian_closed_rk4_step(
                effective_m1,
                effective_m2,
                DT,
            )
        )

        if (
            not np.all(np.isfinite(x))
            or np.max(np.abs(x)) > 1e6
        ):
            print()
            print("Microscopic integration became unstable.")
            break

    # -----------------------------------------------------------------
    # Convert results
    # -----------------------------------------------------------------

    times = np.asarray(times)
    exact_m1_values = np.asarray(exact_m1_values)
    exact_m2_values = np.asarray(exact_m2_values)
    exact_m5_values = np.asarray(exact_m5_values)

    closure_m1_values = np.asarray(closure_m1_values)
    closure_m2_values = np.asarray(closure_m2_values)
    closure_m5_values = np.asarray(closure_m5_values)

    closure_errors = np.asarray(closure_errors)
    stabilising_exact = np.asarray(stabilising_exact)
    stabilising_closure = np.asarray(stabilising_closure)
    max_abs_x = np.asarray(max_abs_x)

    # -----------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------

    max_closure_error = np.max(closure_errors)
    final_closure_error = closure_errors[-1]

    max_m5_error = np.max(
        np.abs(exact_m5_values - closure_m5_values)
    )

    final_m5_error = abs(
        exact_m5_values[-1] - closure_m5_values[-1]
    )

    max_abs_stabilising_difference = np.max(
        np.abs(
            stabilising_exact
            - stabilising_closure
        )
    )

    max_micro_amplitude = np.max(max_abs_x)

    # -----------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------

    data = np.column_stack(
        [
            times,
            exact_m1_values,
            closure_m1_values,
            exact_m2_values,
            closure_m2_values,
            exact_m5_values,
            closure_m5_values,
            closure_errors,
            stabilising_exact,
            stabilising_closure,
            max_abs_x,
        ]
    )

    header = (
        "time,"
        "exact_m1,closure_m1,"
        "exact_m2,closure_m2,"
        "exact_m5,closure_m5,"
        "closure_error,"
        "stabilising_exact,"
        "stabilising_closure,"
        "max_abs_x"
    )

    np.savetxt(
        CSV_PATH,
        data,
        delimiter=",",
        header=header,
        comments="",
    )

    # -----------------------------------------------------------------
    # Save summary
    # -----------------------------------------------------------------

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:

        f.write(
            "Experiment 006f: Principled Closure Breakdown\n\n"
        )

        f.write(
            "Microscopic dynamics:\n"
            "    dx/dt = x + x^3 - x^5\n\n"
        )

        f.write(
            "Exact moment hierarchy:\n"
            "    dm_n/dt = n * (m_n + m_(n+2) - m_(n+4))\n\n"
        )

        f.write(
            "Effective closure:\n"
            "    Gaussian closure using only m1 and m2\n\n"
        )

        f.write(
            "Results:\n"
            f"maximum closure error = {max_closure_error:.12e}\n"
            f"final closure error = {final_closure_error:.12e}\n"
            f"maximum m5 closure error = {max_m5_error:.12e}\n"
            f"final m5 closure error = {final_m5_error:.12e}\n"
            f"maximum stabilising-term difference = "
            f"{max_abs_stabilising_difference:.12e}\n"
            f"maximum microscopic |x| = "
            f"{max_micro_amplitude:.12e}\n"
        )

        if failure_time is None:
            f.write(
                "effective closure failure time = none\n"
            )
        else:
            f.write(
                f"effective closure failure time = "
                f"{failure_time:.12e}\n"
            )

        f.write("\n")
        f.write(
            "Interpretation:\n"
            "The Gaussian closure is derived from retained moments "
            "rather than fitted to the trajectory. The key test is "
            "whether the closure remains accurate when the omitted "
            "higher moments become dynamically important.\n"
        )

    # -----------------------------------------------------------------
    # Plot 1: closure error
    # -----------------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        times,
        closure_errors,
        label="Gaussian closure error",
    )

    plt.xlabel("Time")
    plt.ylabel("Low-order state error")
    plt.title("006f: Closure Error")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_ERROR, dpi=160)
    plt.close()

    # -----------------------------------------------------------------
    # Plot 2: stabilising contribution
    # -----------------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        times,
        stabilising_exact,
        label="Exact -m5",
    )

    plt.plot(
        times,
        stabilising_closure,
        "--",
        label="Gaussian -m5",
    )

    plt.xlabel("Time")
    plt.ylabel("Stabilising contribution")
    plt.title(
        "006f: Exact vs Gaussian Stabilising Contribution"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_STABILISING, dpi=160)
    plt.close()

    # -----------------------------------------------------------------
    # Plot 3: moment comparison
    # -----------------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        times,
        exact_m1_values,
        label="Exact m1",
    )

    plt.plot(
        times,
        closure_m1_values,
        "--",
        label="Gaussian m1",
    )

    plt.plot(
        times,
        exact_m2_values,
        label="Exact m2",
    )

    plt.plot(
        times,
        closure_m2_values,
        "--",
        label="Gaussian m2",
    )

    plt.xlabel("Time")
    plt.ylabel("Moment value")
    plt.title("006f: Exact vs Closed Moments")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_MOMENTS, dpi=160)
    plt.close()

    # -----------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------

    print("Results:")
    print(
        f"maximum closure error = "
        f"{max_closure_error:.12e}"
    )

    print(
        f"final closure error = "
        f"{final_closure_error:.12e}"
    )

    print(
        f"maximum m5 closure error = "
        f"{max_m5_error:.12e}"
    )

    print(
        f"final m5 closure error = "
        f"{final_m5_error:.12e}"
    )

    print(
        f"maximum stabilising-term difference = "
        f"{max_abs_stabilising_difference:.12e}"
    )

    print(
        f"maximum microscopic |x| = "
        f"{max_micro_amplitude:.12e}"
    )

    if failure_time is None:
        print(
            "effective closure failure time = none"
        )
    else:
        print(
            f"effective closure failure time = "
            f"{failure_time:.12e}"
        )

    print()
    print("Outputs:")
    print(f"    {CSV_PATH}")
    print(f"    {SUMMARY_PATH}")
    print(f"    {PLOT_ERROR}")
    print(f"    {PLOT_STABILISING}")
    print(f"    {PLOT_MOMENTS}")


if __name__ == "__main__":
    main()
