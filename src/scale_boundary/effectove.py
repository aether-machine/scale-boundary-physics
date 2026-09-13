"""
Effective / coarse-scale dynamical models.

This module contains the dynamics used after a microscopic state has been
coarse-grained.

The key methodological distinction is:

    Route A:
        microscopic evolution -> coarse-graining

    Route B:
        coarse-graining -> independent effective evolution

The effective model therefore receives only the coarse state as its
initial condition. It has no access to the microscopic variables that
were discarded by the coarse-graining map.

For Experiment 001, the effective model uses the same general oscillator
structure as the microscopic model, but operates on the reduced lattice.

This is intentionally a simple closure. Later experiments can replace it
with progressively more physically motivated effective equations.
"""

from __future__ import annotations

import numpy as np


def acceleration(
    Q: np.ndarray,
    k: float = 1.0,
    omega0_sq: float = 0.5,
    beta: float = 0.25,
) -> np.ndarray:
    """
    Calculate acceleration on the coarse lattice.

    Parameters
    ----------
    Q:
        Coarse displacement variables.
    k:
        Effective nearest-neighbour coupling strength.
    omega0_sq:
        Effective on-site restoring coefficient.
    beta:
        Effective nonlinear coefficient.

    Returns
    -------
    np.ndarray
        Coarse acceleration.

    Notes
    -----
    The parameters are currently inherited from the microscopic model.

    This is a deliberate simplification for Experiment 001. In a more
    rigorous effective theory, these parameters should generally be
    derived from the microscopic dynamics and the coarse-graining map.

    In particular, future experiments should investigate whether the
    effective coefficients become scale-dependent.
    """

    neighbour_term = (
        2.0 * Q
        - np.roll(Q, 1)
        - np.roll(Q, -1)
    )

    return (
        -k * neighbour_term
        - omega0_sq * Q
        - beta * Q**3
    )


def rhs(
    Q: np.ndarray,
    P: np.ndarray,
    k: float = 1.0,
    omega0_sq: float = 0.5,
    beta: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the effective first-order dynamical system.

    The effective state is

        H = (Q, P)

    and evolves independently of the microscopic state.

    Returns
    -------
    dQdt, dPdt
    """

    dQdt = P

    dPdt = acceleration(
        Q,
        k=k,
        omega0_sq=omega0_sq,
        beta=beta,
    )

    return dQdt, dPdt


def rk4_step(
    Q: np.ndarray,
    P: np.ndarray,
    dt: float,
    k: float = 1.0,
    omega0_sq: float = 0.5,
    beta: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Advance the effective system by one fourth-order Runge-Kutta step.

    This function deliberately operates only on the coarse variables.

    No microscopic information is accessed during the evolution.
    """

    k1_Q, k1_P = rhs(
        Q,
        P,
        k=k,
        omega0_sq=omega0_sq,
        beta=beta,
    )

    k2_Q, k2_P = rhs(
        Q + 0.5 * dt * k1_Q,
        P + 0.5 * dt * k1_P,
        k=k,
        omega0_sq=omega0_sq,
        beta=beta,
    )

    k3_Q, k3_P = rhs(
        Q + 0.5 * dt * k2_Q,
        P + 0.5 * dt * k2_P,
        k=k,
        omega0_sq=omega0_sq,
        beta=beta,
    )

    k4_Q, k4_P = rhs(
        Q + dt * k3_Q,
        P + dt * k3_P,
        k=k,
        omega0_sq=omega0_sq,
        beta=beta,
    )

    Q_new = Q + (dt / 6.0) * (
        k1_Q
        + 2.0 * k2_Q
        + 2.0 * k3_Q
        + k4_Q
    )

    P_new = P + (dt / 6.0) * (
        k1_P
        + 2.0 * k2_P
        + 2.0 * k3_P
        + k4_P
    )

    return Q_new, P_new
