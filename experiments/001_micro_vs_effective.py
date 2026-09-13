"""
Experiment 001: Microscopic dynamics vs. effective dynamics

Purpose
-------
Test whether coarse-graining and dynamical evolution approximately commute.

We compare two independently evolved descriptions of the same initial system:

    Route A: microscopic evolution, then coarse-graining

        X0 -> Phi_micro(t) -> X(t) -> C[X(t)]

    Route B: coarse-grain first, then evolve an effective model

        X0 -> C[X0] -> H0 -> Phi_eff(t) -> H(t)

The central diagnostic is

    Delta(t) = C[X(t)] - H(t)

where Delta measures the failure of the two routes to agree.

This is deliberately a toy model. It is NOT intended to model Navier-Stokes
directly. Its purpose is methodological: establish a clean computational
framework in which scale transitions can be measured before introducing
fluid dynamics or a proposed underlying-medium theory.

Microscopic model
-----------------
A periodic one-dimensional lattice of nonlinear oscillators:

    d q_i / dt = p_i

    d p_i / dt = -k (2 q_i - q_{i-1} - q_{i+1})
                 - omega0^2 q_i
                 - beta q_i^3

The cubic term makes the microscopic dynamics nonlinear.

Coarse-graining
---------------
The microscopic lattice is divided into blocks.

For each block we retain:

    Q = mean(q_i)
    P = mean(p_i)

The effective model evolves those block variables using the same structural
form of the lattice equation, but on the reduced coarse lattice.

Important:
The effective model is NOT obtained by continuing the microscopic trajectory.
It is evolved independently from the coarse initial condition.

Outputs
-------
The script produces:

    results/experiment_001_delta.png
    results/experiment_001_states.png
    results/experiment_001_summary.txt

The first plot shows the coarse-graining/evolution discrepancy.

The second plot compares one representative coarse variable obtained by
the two routes.

The experiment also prints a summary of the discrepancy.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_MICRO = 128
BLOCK_SIZE = 8
N_COARSE = N_MICRO // BLOCK_SIZE

DT = 0.002
T_FINAL = 20.0

K = 1.0
OMEGA0_SQ = 0.5
BETA = 0.25

# Initial condition parameters
AMPLITUDE = 0.8
WAVELENGTHS = 4

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "results"


# ---------------------------------------------------------------------------
# Microscopic model
# ---------------------------------------------------------------------------

def acceleration(q: np.ndarray, k: float, omega0_sq: float, beta: float) -> np.ndarray:
    """
    Calculate acceleration for a periodic nonlinear oscillator lattice.

    Periodic boundary conditions are implemented with np.roll().
    """
    neighbour_term = 2.0 * q - np.roll(q, 1) - np.roll(q, -1)

    return (
        -k * neighbour_term
        - omega0_sq * q
        - beta * q**3
    )


def rhs(
    q: np.ndarray,
    p: np.ndarray,
    k: float,
    omega0_sq: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return dq/dt and dp/dt."""
    dqdt = p
    dpdt = acceleration(q, k, omega0_sq, beta)

    return dqdt, dpdt


# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------

def rk4_step(
    q: np.ndarray,
    p: np.ndarray,
    dt: float,
    k: float,
    omega0_sq: float,
    beta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One fourth-order Runge-Kutta step.

    Both microscopic and effective systems use the same integrator, but
    they are evolved independently.
    """

    k1_q, k1_p = rhs(q, p, k, omega0_sq, beta)

    k2_q, k2_p = rhs(
        q + 0.5 * dt * k1_q,
        p + 0.5 * dt * k1_p,
        k,
        omega0_sq,
        beta,
    )

    k3_q, k3_p = rhs(
        q + 0.5 * dt * k2_q,
        p + 0.5 * dt * k2_p,
        k,
        omega0_sq,
        beta,
    )

    k4_q, k4_p = rhs(
        q + dt * k3_q,
        p + dt * k3_p,
        k,
        omega0_sq,
        beta,
    )

    q_new = q + (dt / 6.0) * (
        k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q
    )

    p_new = p + (dt / 6.0) * (
        k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p
    )

    return q_new, p_new


# ---------------------------------------------------------------------------
# Coarse-graining
# ---------------------------------------------------------------------------

def coarse_grain(
    q: np.ndarray,
    p: np.ndarray,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Coarse-grain the microscopic state by block averaging.

    The microscopic lattice is reshaped into blocks:

        [block 0]
        [block 1]
        ...

    and the mean q and p values are retained.

    This deliberately discards all intra-block structure.
    """

    n = len(q)

    if n % block_size != 0:
        raise ValueError("Number of microscopic sites must divide evenly.")

    n_blocks = n // block_size

    q_blocks = q.reshape(n_blocks, block_size)
    p_blocks = p.reshape(n_blocks, block_size)

    Q = q_blocks.mean(axis=1)
    P = p_blocks.mean(axis=1)

    return Q, P


# ---------------------------------------------------------------------------
# Initial condition
# ---------------------------------------------------------------------------

def initial_condition(n: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct a smooth microscopic initial condition.

    A low-frequency wave is used so that the initial coarse description
    captures most of the visible structure.

    A small higher-frequency perturbation is added. This is important:
    the microscopic state contains information that the coarse state does
    not retain.
    """

    x = np.arange(n)

    low_mode = AMPLITUDE * np.sin(
        2.0 * math.pi * WAVELENGTHS * x / n
    )

    high_mode = 0.15 * np.sin(
        2.0 * math.pi * (WAVELENGTHS * 8) * x / n
    )

    q0 = low_mode + high_mode

    # Initial momentum is chosen to be zero.
    p0 = np.zeros_like(q0)

    return q0, p0


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_experiment():
    """
    Run the two routes independently.

    Route A
    -------
        microscopic evolution -> coarse-graining

    Route B
    -------
        coarse initial condition -> independent effective evolution
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    q_micro, p_micro = initial_condition(N_MICRO)

    # ---------------------------------------------------------------
    # Construct the coarse initial state ONCE.
    #
    # This is the only information Route B receives from the
    # microscopic initial condition.
    # ---------------------------------------------------------------

    Q0, P0 = coarse_grain(
        q_micro,
        p_micro,
        BLOCK_SIZE,
    )

    # ---------------------------------------------------------------
    # Effective system.
    #
    # It has N_COARSE degrees of freedom and evolves independently.
    # ---------------------------------------------------------------

    Q_eff = Q0.copy()
    P_eff = P0.copy()

    times = []
    delta_q_norm = []
    delta_p_norm = []
    delta_total = []

    micro_history = []
    effective_history = []

    n_steps = int(round(T_FINAL / DT))

    sample_every = max(1, n_steps // 1000)

    for step in range(n_steps + 1):

        t = step * DT

        # -----------------------------------------------------------
        # Route A:
        #
        # Continue microscopic evolution.
        # Then coarse-grain the resulting microscopic state.
        # -----------------------------------------------------------

        Q_from_micro, P_from_micro = coarse_grain(
            q_micro,
            p_micro,
            BLOCK_SIZE,
        )

        # -----------------------------------------------------------
        # Route B:
        #
        # Q_eff and P_eff have been evolving independently.
        # -----------------------------------------------------------

        if step % sample_every == 0:

            # -------------------------------------------------------
            # Difference between the two descriptions.
            # -------------------------------------------------------

            dq = Q_from_micro - Q_eff
            dp = P_from_micro - P_eff

            q_error = np.linalg.norm(dq) / math.sqrt(N_COARSE)
            p_error = np.linalg.norm(dp) / math.sqrt(N_COARSE)

            total_error = math.sqrt(
                q_error**2 + p_error**2
            )

            times.append(t)
            delta_q_norm.append(q_error)
            delta_p_norm.append(p_error)
            delta_total.append(total_error)

            # Save states for later visualization.
            micro_history.append(Q_from_micro.copy())
            effective_history.append(Q_eff.copy())

        # -----------------------------------------------------------
        # Advance BOTH systems.
        #
        # These are separate calls with separate state variables.
        # There is no feedback from Route A into Route B.
        # -----------------------------------------------------------

        if step < n_steps:

            q_micro, p_micro = rk4_step(
                q_micro,
                p_micro,
                DT,
                K,
                OMEGA0_SQ,
                BETA,
            )

            Q_eff, P_eff = rk4_step(
                Q_eff,
                P_eff,
                DT,
                K,
                OMEGA0_SQ,
                BETA,
            )

    times = np.asarray(times)
    delta_q_norm = np.asarray(delta_q_norm)
    delta_p_norm = np.asarray(delta_p_norm)
    delta_total = np.asarray(delta_total)

    micro_history = np.asarray(micro_history)
    effective_history = np.asarray(effective_history)

    # ----------------------------------------------------------------
    # Diagnostics
    # ----------------------------------------------------------------

    max_error = float(np.max(delta_total))
    final_error = float(delta_total[-1])

    peak_time = float(times[np.argmax(delta_total)])

    initial_error = float(delta_total[0])

    summary = f"""
Experiment 001: Microscopic vs. Effective Dynamics
====================================================

Microscopic sites:       {N_MICRO}
Coarse blocks:            {N_COARSE}
Block size:               {BLOCK_SIZE}

Time step:                {DT}
Final time:               {T_FINAL}

Model parameters
----------------
k:                        {K}
omega0^2:                 {OMEGA0_SQ}
beta:                     {BETA}

Discrepancy
-----------
Initial Delta norm:       {initial_error:.8e}
Final Delta norm:         {final_error:.8e}
Maximum Delta norm:       {max_error:.8e}
Time of maximum Delta:    {peak_time:.6f}

Interpretation
--------------
The two routes begin from the same microscopic initial condition,
but Route B only receives the block-averaged initial state.

Therefore any subsequent disagreement is information that is available
to the microscopic dynamics but absent from the coarse representation.

This experiment does NOT establish that a physical singularity is caused
by coarse-graining. It establishes a computational framework for measuring
the non-commutativity of coarse-graining and dynamical evolution.

Next steps should investigate whether the discrepancy correlates with
a physically meaningful scale parameter such as wavelength, block size,
or an analogue of the Knudsen number.
"""

    print(summary)

    with open(
        OUTPUT_DIR / "experiment_001_summary.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(summary)

    # ----------------------------------------------------------------
    # Plot 1: discrepancy
    # ----------------------------------------------------------------

    plt.figure(figsize=(9, 5))

    plt.plot(
        times,
        delta_total,
        label=r"$\|\Delta(t)\|$",
    )

    plt.plot(
        times,
        delta_q_norm,
        label=r"$\|\Delta_Q(t)\|$",
    )

    plt.plot(
        times,
        delta_p_norm,
        label=r"$\|\Delta_P(t)\|$",
    )

    plt.xlabel("Time")
    plt.ylabel("RMS discrepancy")
    plt.title("Failure of coarse-graining and evolution to commute")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "experiment_001_delta.png",
        dpi=150,
    )

    plt.close()

    # ----------------------------------------------------------------
    # Plot 2: representative coarse variable
    # ----------------------------------------------------------------

    representative_block = 0

    plt.figure(figsize=(9, 5))

    plt.plot(
        times,
        micro_history[:, representative_block],
        label="Route A: microscopic → coarse",
    )

    plt.plot(
        times,
        effective_history[:, representative_block],
        "--",
        label="Route B: coarse → effective",
    )

    plt.xlabel("Time")
    plt.ylabel("Coarse displacement")
    plt.title(
        f"Representative coarse variable "
        f"(block {representative_block})"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "experiment_001_states.png",
        dpi=150,
    )

    plt.close()

    return {
        "times": times,
        "delta_total": delta_total,
        "delta_q": delta_q_norm,
        "delta_p": delta_p_norm,
    }


if __name__ == "__main__":
    run_experiment()
