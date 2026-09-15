"""
Experiment 006h: Memory-Augmented Closure

Question:
Can a small amount of dynamical memory restore predictive closure
when instantaneous coarse variables are insufficient?

Microscopic dynamics:

    dx/dt = x + x^3 - x^5

Retained variables:

    m1 = E[x]
    m2 = E[x^2]

From experiment 006g we know that two microscopic ensembles can have
the same (m1, m2) while possessing different future derivatives.

Here we test whether augmenting the retained state with a memory
variable can recover some of that missing predictive information.

This is deliberately a toy memory model.

The memory variable obeys

    dM/dt = -M/tau_M + K * H

and contributes to the effective dynamics.

We compare:

    A. Instantaneous closure
    B. Memory-augmented closure
    C. Exact microscopic evolution

IMPORTANT:
The memory model is not claimed to be a physical model of fluids.
It is a test of the general proposition that information discarded
by coarse-graining may reappear as temporal memory rather than as
additional instantaneous state variables.

Outputs:

    results/006h_memory_closure.csv
    results/006h_memory_closure_summary.txt
    results/006h_memory_prediction.png
    results/006h_memory_error.png
    results/006h_memory_variable.png
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

CSV_PATH = (
    OUTPUT_DIR
    / "006h_memory_closure.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "006h_memory_closure_summary.txt"
)

PLOT_PREDICTION = (
    OUTPUT_DIR
    / "006h_memory_prediction.png"
)

PLOT_ERROR = (
    OUTPUT_DIR
    / "006h_memory_error.png"
)

PLOT_MEMORY = (
    OUTPUT_DIR
    / "006h_memory_variable.png"
)


# ---------------------------------------------------------------------
# Microscopic dynamics
# ---------------------------------------------------------------------

def microscopic_rhs(x):
    """
    dx/dt = x + x^3 - x^5
    """
    return (
        x
        + x**3
        - x**5
    )


def rk4_step(x, dt):
    """
    RK4 integration of the microscopic ensemble.
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
# Moment calculations
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
    Exact hierarchy:

        dm_n/dt =
            n * (m_n + m_(n+2) - m_(n+4))
    """

    return n * (
        m[n]
        + m[n + 2]
        - m[n + 4]
    )


# ---------------------------------------------------------------------
# Ensemble construction
# ---------------------------------------------------------------------

def make_ensembles():
    """
    Construct two ensembles with identical m1 and m2 but different
    higher-order structure.

    Ensemble A:
        approximately Gaussian

    Ensemble B:
        three-component mixture

    Both are affine-normalised to match the mean and variance of A.
    """

    rng = np.random.default_rng(12345)

    # Ensemble A
    x_a = rng.normal(
        loc=1.0,
        scale=0.20,
        size=N,
    )

    # Ensemble B
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

    target_mean = np.mean(x_a)
    target_variance = np.var(
        x_a,
        ddof=0,
    )

    b_mean = np.mean(x_b)
    b_std = np.std(x_b)

    x_b = (
        (x_b - b_mean)
        / b_std
        * np.sqrt(target_variance)
        + target_mean
    )

    return x_a, x_b


# ---------------------------------------------------------------------
# Instantaneous closure
# ---------------------------------------------------------------------

def instantaneous_closure_rhs(
    m1,
    m2,
):
    """
    Simple Gaussian instantaneous closure.

    Higher moments are reconstructed from m1 and m2.

    This is intentionally the same basic closure tested in 006f.
    """

    variance = max(
        m2 - m1**2,
        0.0,
    )

    # Gaussian raw moments needed here.

    m3 = (
        m1**3
        + 3.0 * m1 * variance
    )

    m4 = (
        m1**4
        + 6.0 * m1**2 * variance
        + 3.0 * variance**2
    )

    m5 = (
        m1**5
        + 10.0 * m1**3 * variance
        + 15.0 * m1 * variance**2
    )

    m6 = (
        m1**6
        + 15.0 * m1**4 * variance
        + 45.0 * m1**2 * variance**2
        + 15.0 * variance**3
    )

    dm1 = (
        m1
        + m3
        - m5
    )

    dm2 = 2.0 * (
        m2
        + m4
        - m6
    )

    return dm1, dm2


# ---------------------------------------------------------------------
# Memory-augmented closure
# ---------------------------------------------------------------------

def memory_closure_rhs(
    m1,
    m2,
    memory,
    tau_memory,
    coupling,
):
    """
    Memory-augmented effective dynamics.

    The instantaneous part uses the Gaussian closure.

    The memory variable represents a simple exponentially decaying
    record of unresolved dynamics.

        dM/dt = -M/tau_memory + coupling * H

    The memory contribution is intentionally simple and linear.

    H is represented by the two retained variables.
    """

    dm1_base, dm2_base = (
        instantaneous_closure_rhs(
            m1,
            m2,
        )
    )

    # Memory contribution.
    #
    # The signs are chosen so that positive memory tends to oppose
    # rapid excursions in the effective state.

    memory_m1 = memory[0]
    memory_m2 = memory[1]

    dm1 = (
        dm1_base
        - memory_m1
    )

    dm2 = (
        dm2_base
        - memory_m2
    )

    dmemory_1 = (
        -memory_m1 / tau_memory
        + coupling * dm1_base
    )

    dmemory_2 = (
        -memory_m2 / tau_memory
        + coupling * dm2_base
    )

    return (
        dm1,
        dm2,
        dmemory_1,
        dmemory_2,
    )


# ---------------------------------------------------------------------
# RK4 step for memory closure
# ---------------------------------------------------------------------

def memory_rk4_step(
    state,
    dt,
    tau_memory,
    coupling,
):
    """
    RK4 integration of

        state =
            [m1, m2, M1, M2].
    """

    def rhs(s):

        m1 = s[0]
        m2 = s[1]

        memory = s[2:4]

        values = memory_closure_rhs(
            m1,
            m2,
            memory,
            tau_memory,
            coupling,
        )

        return np.array(
            values,
            dtype=float,
        )

    k1 = rhs(state)

    k2 = rhs(
        state
        + 0.5 * dt * k1
    )

    k3 = rhs(
        state
        + 0.5 * dt * k2
    )

    k4 = rhs(
        state
        + dt * k3
    )

    return state + (
        dt / 6.0
    ) * (
        k1
        + 2.0 * k2
        + 2.0 * k3
        + k4
    )


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():

    print(
        "Experiment 006h: Memory-Augmented Closure"
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
        "Effective models:"
    )
    print(
        "    1. instantaneous Gaussian closure"
    )
    print(
        "    2. memory-augmented closure"
    )
    print()

    # -----------------------------------------------------------------
    # Initial microscopic states
    # -----------------------------------------------------------------

    x_a, x_b = make_ensembles()

    m_a = moments(
        x_a,
        max_order=8,
    )

    m_b = moments(
        x_b,
        max_order=8,
    )

    initial_m1 = m_a[1]
    initial_m2 = m_a[2]

    print(
        "Initial retained state:"
    )

    print(
        f"    m1 = "
        f"{initial_m1:.15e}"
    )

    print(
        f"    m2 = "
        f"{initial_m2:.15e}"
    )

    print()

    # -----------------------------------------------------------------
    # Memory parameters
    # -----------------------------------------------------------------

    tau_memory = 0.25
    coupling = 0.50

    print(
        "Memory parameters:"
    )

    print(
        f"    tau_memory = "
        f"{tau_memory:.6f}"
    )

    print(
        f"    coupling = "
        f"{coupling:.6f}"
    )

    print()

    # -----------------------------------------------------------------
    # Effective states
    # -----------------------------------------------------------------

    instantaneous = np.array(
        [
            initial_m1,
            initial_m2,
        ],
        dtype=float,
    )

    memory_state = np.array(
        [
            initial_m1,
            initial_m2,
            0.0,
            0.0,
        ],
        dtype=float,
    )

    # -----------------------------------------------------------------
    # Storage
    # -----------------------------------------------------------------

    times = []

    exact_m1 = []
    exact_m2 = []

    instantaneous_m1 = []
    instantaneous_m2 = []

    memory_m1 = []
    memory_m2 = []

    memory_1 = []
    memory_2 = []

    instantaneous_error = []
    memory_error = []

    exact_hidden = []
    memory_magnitude = []

    steps = int(
        T / DT
    )

    for step in range(
        steps + 1
    ):

        t = step * DT

        # -------------------------------------------------------------
        # Exact microscopic moments
        # -------------------------------------------------------------

        exact = moments(
            x_a,
            max_order=8,
        )

        exact_state = np.array(
            [
                exact[1],
                exact[2],
            ]
        )

        # -------------------------------------------------------------
        # Effective states
        # -------------------------------------------------------------

        inst_state = instantaneous

        mem_state = memory_state[:2]

        # -------------------------------------------------------------
        # Errors
        # -------------------------------------------------------------

        inst_error = np.linalg.norm(
            exact_state
            - inst_state
        )

        mem_error = np.linalg.norm(
            exact_state
            - mem_state
        )

        # -------------------------------------------------------------
        # Hidden moment magnitude
        # -------------------------------------------------------------

        hidden = np.sqrt(
            exact[3]**2
            + exact[4]**2
            + exact[5]**2
        )

        mem_mag = np.linalg.norm(
            memory_state[2:4]
        )

        # -------------------------------------------------------------
        # Store
        # -------------------------------------------------------------

        times.append(t)

        exact_m1.append(
            exact[1]
        )

        exact_m2.append(
            exact[2]
        )

        instantaneous_m1.append(
            instantaneous[0]
        )

        instantaneous_m2.append(
            instantaneous[1]
        )

        memory_m1.append(
            memory_state[0]
        )

        memory_m2.append(
            memory_state[1]
        )

        memory_1.append(
            memory_state[2]
        )

        memory_2.append(
            memory_state[3]
        )

        instantaneous_error.append(
            inst_error
        )

        memory_error.append(
            mem_error
        )

        exact_hidden.append(
            hidden
        )

        memory_magnitude.append(
            mem_mag
        )

        if step == steps:
            break

        # -------------------------------------------------------------
        # Advance microscopic system A
        # -------------------------------------------------------------

        x_a = rk4_step(
            x_a,
            DT,
        )

        # -------------------------------------------------------------
        # Advance instantaneous closure
        # -------------------------------------------------------------

        dm1, dm2 = (
            instantaneous_closure_rhs(
                instantaneous[0],
                instantaneous[1],
            )
        )

        instantaneous = (
            instantaneous
            + DT
            * np.array(
                [
                    dm1,
                    dm2,
                ]
            )
        )

        # -------------------------------------------------------------
        # Advance memory closure
        # -------------------------------------------------------------

        memory_state = (
            memory_rk4_step(
                memory_state,
                DT,
                tau_memory,
                coupling,
            )
        )

        # -------------------------------------------------------------
        # Numerical safety
        # -------------------------------------------------------------

        if (
            not np.all(
                np.isfinite(x_a)
            )
            or
            not np.all(
                np.isfinite(
                    instantaneous
                )
            )
            or
            not np.all(
                np.isfinite(
                    memory_state
                )
            )
        ):
            print(
                "Numerical failure detected."
            )
            break

    # -----------------------------------------------------------------
    # Convert arrays
    # -----------------------------------------------------------------

    times = np.asarray(times)

    exact_m1 = np.asarray(
        exact_m1
    )
    exact_m2 = np.asarray(
        exact_m2
    )

    instantaneous_m1 = np.asarray(
        instantaneous_m1
    )
    instantaneous_m2 = np.asarray(
        instantaneous_m2
    )

    memory_m1 = np.asarray(
        memory_m1
    )
    memory_m2 = np.asarray(
        memory_m2
    )

    memory_1 = np.asarray(
        memory_1
    )
    memory_2 = np.asarray(
        memory_2
    )

    instantaneous_error = np.asarray(
        instantaneous_error
    )

    memory_error = np.asarray(
        memory_error
    )

    exact_hidden = np.asarray(
        exact_hidden
    )

    memory_magnitude = np.asarray(
        memory_magnitude
    )

    # -----------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------

    max_inst_error = np.max(
        instantaneous_error
    )

    final_inst_error = (
        instantaneous_error[-1]
    )

    max_memory_error = np.max(
        memory_error
    )

    final_memory_error = (
        memory_error[-1]
    )

    improvement = (
        max_inst_error
        - max_memory_error
    )

    improvement_fraction = (
        improvement
        / max_inst_error
        if max_inst_error > 0
        else 0.0
    )

    # -----------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------

    data = np.column_stack(
        [
            times,

            exact_m1,
            exact_m2,

            instantaneous_m1,
            instantaneous_m2,

            memory_m1,
            memory_m2,

            memory_1,
            memory_2,

            instantaneous_error,
            memory_error,

            exact_hidden,
            memory_magnitude,
        ]
    )

    header = (
        "time,"
        "exact_m1,exact_m2,"
        "instantaneous_m1,instantaneous_m2,"
        "memory_m1,memory_m2,"
        "memory_1,memory_2,"
        "instantaneous_error,"
        "memory_error,"
        "exact_hidden,"
        "memory_magnitude"
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

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Experiment 006h: "
            "Memory-Augmented Closure\n\n"
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
            "Memory model:\n"
            f"    tau_memory = "
            f"{tau_memory}\n"
            f"    coupling = "
            f"{coupling}\n\n"
        )

        f.write(
            "Results:\n"
            f"maximum instantaneous error = "
            f"{max_inst_error:.12e}\n"
            f"final instantaneous error = "
            f"{final_inst_error:.12e}\n"
            f"maximum memory-closure error = "
            f"{max_memory_error:.12e}\n"
            f"final memory-closure error = "
            f"{final_memory_error:.12e}\n"
            f"maximum error reduction = "
            f"{improvement:.12e}\n"
            f"fractional maximum error reduction = "
            f"{improvement_fraction:.12e}\n"
        )

        f.write(
            "\nInterpretation:\n"
        )

        f.write(
            "The instantaneous closure uses only the current "
            "retained variables. The memory closure augments "
            "them with a small dynamical state representing "
            "history from discarded degrees of freedom.\n"
        )

        f.write(
            "A reduction in prediction error would demonstrate "
            "that temporal memory can partially restore information "
            "lost through coarse-graining.\n"
        )

        f.write(
            "The particular memory equation is a toy model and "
            "should not be interpreted as a physical derivation.\n"
        )

    # -----------------------------------------------------------------
    # Plot 1: predictions
    # -----------------------------------------------------------------

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times,
        exact_m1,
        label="Exact m1",
    )

    plt.plot(
        times,
        instantaneous_m1,
        "--",
        label="Instantaneous closure",
    )

    plt.plot(
        times,
        memory_m1,
        ":",
        label="Memory closure",
    )

    plt.xlabel("Time")
    plt.ylabel("m1")
    plt.title(
        "006h: Exact vs Effective Prediction"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOT_PREDICTION,
        dpi=160,
    )

    plt.close()

    # -----------------------------------------------------------------
    # Plot 2: errors
    # -----------------------------------------------------------------

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times,
        instantaneous_error,
        label="Instantaneous closure error",
    )

    plt.plot(
        times,
        memory_error,
        label="Memory closure error",
    )

    plt.xlabel("Time")
    plt.ylabel("Prediction error")
    plt.title(
        "006h: Does Memory Improve Closure?"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOT_ERROR,
        dpi=160,
    )

    plt.close()

    # -----------------------------------------------------------------
    # Plot 3: memory variable
    # -----------------------------------------------------------------

    plt.figure(
        figsize=(9, 5)
    )

    plt.plot(
        times,
        memory_1,
        label="Memory M1",
    )

    plt.plot(
        times,
        memory_2,
        label="Memory M2",
    )

    plt.plot(
        times,
        exact_hidden,
        "--",
        label="Hidden-state magnitude",
    )

    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.title(
        "006h: Effective Memory vs Hidden Dynamics"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOT_MEMORY,
        dpi=160,
    )

    plt.close()

    # -----------------------------------------------------------------
    # Console summary
    # -----------------------------------------------------------------

    print()
    print("Results:")

    print(
        f"maximum instantaneous error = "
        f"{max_inst_error:.12e}"
    )

    print(
        f"final instantaneous error = "
        f"{final_inst_error:.12e}"
    )

    print(
        f"maximum memory-closure error = "
        f"{max_memory_error:.12e}"
    )

    print(
        f"final memory-closure error = "
        f"{final_memory_error:.12e}"
    )

    print(
        f"maximum error reduction = "
        f"{improvement:.12e}"
    )

    print(
        f"fractional maximum error reduction = "
        f"{improvement_fraction:.12e}"
    )

    print()
    print("Outputs:")

    print(
        f"    {CSV_PATH}"
    )

    print(
        f"    {SUMMARY_PATH}"
    )

    print(
        f"    {PLOT_PREDICTION}"
    )

    print(
        f"    {PLOT_ERROR}"
    )

    print(
        f"    {PLOT_MEMORY}"
    )


if __name__ == "__main__":
    main()
