"""
Microscopic dynamical systems.

This module contains the underlying high-resolution dynamical model used
by the initial experiments.

The current model is a one-dimensional periodic lattice of nonlinear
oscillators:

    dq_i/dt = p_i

    dp_i/dt =
        -k (2q_i - q_{i-1} - q_{i+1})
        - omega0^2 q_i
        - beta q_i^3

The implementation is intentionally general enough that later experiments
can replace the microscopic model without changing the coarse-graining
or diagnostic machinery.
"""

from __future__ import annotations

import numpy as np


def acceleration(
    q: np.ndarray,
    k: float = 1.0,
    omega0_sq: float = 0.5,
    beta: float = 0.25,
) -> np.ndarray:
    """
    Calculate the acceleration of every microscopic oscillator.

    Parameters
    ----------
    q:
        Displacement of each oscillator.
    k:
        Nearest-neighbour coupling strength.
    omega0_sq:
        On-site restoring-force coefficient.
    beta:
        Cubic nonlinear coefficient.

    Returns
    -------
    np.ndarray
        Acceleration dp/dt.
    """

    neighbour_term = (
        2.0 * q
        - np.roll(q, 1)
        - np.roll(q, -1)
    )

    return (
        -k * neighbour_term
        - omega0_sq * q
        - beta * q**3
    )


def rhs(
    q: np.ndarray,
    p: np.ndarray,
    k: float = 1.0,
    omega0_sq: float = 0.5,
    beta: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return the first-order dynamical system.

    Returns
    -------
    dqdt, dpdt
    """

    dqdt = p

    dpdt = acceleration(
        q,
        k=k,
        omega0_sq=omega0_sq,
        beta=beta,
    )

    return dqdt, dpdt
