"""
Metrics for comparing microscopic and effective descriptions.

The central quantity in this project is the discrepancy between two routes:

    Route A:
        X0 -> microscopic evolution -> coarse-graining

    Route B:
        X0 -> coarse-graining -> effective evolution

The fundamental discrepancy is therefore

    Delta(t) =
        C[Phi_micro(t, X0)]
        - Phi_eff(t, C[X0])

This module provides reusable functions for calculating and summarizing
that discrepancy.

The metrics are deliberately representation-agnostic where possible so
that later experiments can use them with oscillator lattices, kinetic
models, fluid variables, or other state representations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Basic vector metrics
# ---------------------------------------------------------------------------

def l2_norm(values: np.ndarray) -> float:
    """
    Calculate the Euclidean L2 norm of an array.

    Parameters
    ----------
    values:
        Array representing a state or state difference.

    Returns
    -------
    float
        Euclidean norm.
    """

    values = np.asarray(values, dtype=float)

    return float(np.linalg.norm(values))


def rms(values: np.ndarray) -> float:
    """
    Calculate the root-mean-square magnitude.

    Unlike the ordinary L2 norm, RMS does not grow simply because the
    number of degrees of freedom increases.

    This makes it useful when comparing systems at different resolutions.
    """

    values = np.asarray(values, dtype=float)

    return float(np.sqrt(np.mean(values**2)))


def relative_error(
    reference: np.ndarray,
    approximation: np.ndarray,
    floor: float = 1e-12,
) -> float:
    """
    Calculate relative RMS error.

        RMS(reference - approximation)
        --------------------------------
               RMS(reference)

    A small floor prevents division by zero when the reference state
    is identically zero.
    """

    reference = np.asarray(reference, dtype=float)
    approximation = np.asarray(approximation, dtype=float)

    if reference.shape != approximation.shape:
        raise ValueError(
            "reference and approximation must have the same shape."
        )

    numerator = rms(reference - approximation)
    denominator = max(rms(reference), floor)

    return numerator / denominator


# ---------------------------------------------------------------------------
# State discrepancy
# ---------------------------------------------------------------------------

def state_difference(
    reference_q: np.ndarray,
    reference_p: np.ndarray,
    effective_q: np.ndarray,
    effective_p: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the component-wise discrepancy between two coarse states.

    Returns
    -------
    delta_q, delta_p
        Differences in displacement and momentum.
    """

    reference_q = np.asarray(reference_q, dtype=float)
    reference_p = np.asarray(reference_p, dtype=float)

    effective_q = np.asarray(effective_q, dtype=float)
    effective_p = np.asarray(effective_p, dtype=float)

    if reference_q.shape != effective_q.shape:
        raise ValueError(
            "Displacement arrays must have matching shapes."
        )

    if reference_p.shape != effective_p.shape:
        raise ValueError(
            "Momentum arrays must have matching shapes."
        )

    delta_q = reference_q - effective_q
    delta_p = reference_p - effective_p

    return delta_q, delta_p


def state_discrepancy(
    reference_q: np.ndarray,
    reference_p: np.ndarray,
    effective_q: np.ndarray,
    effective_p: np.ndarray,
) -> tuple[float, float, float]:
    """
    Calculate RMS discrepancies for q, p, and the combined state.

    The combined discrepancy is

        Delta = sqrt(Delta_q^2 + Delta_p^2)

    Returns
    -------
    delta_q_rms:
        RMS displacement discrepancy.

    delta_p_rms:
        RMS momentum discrepancy.

    delta_total:
        Combined state discrepancy.
    """

    delta_q, delta_p = state_difference(
        reference_q,
        reference_p,
        effective_q,
        effective_p,
    )

    delta_q_rms = rms(delta_q)
    delta_p_rms = rms(delta_p)

    delta_total = float(
        np.sqrt(
            delta_q_rms**2
            + delta_p_rms**2
        )
    )

    return (
        delta_q_rms,
        delta_p_rms,
        delta_total,
    )


# ---------------------------------------------------------------------------
# Time-series summary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiscrepancySummary:
    """
    Summary statistics for a discrepancy time series.
    """

    initial: float
    final: float
    maximum: float
    mean: float
    time_of_maximum: float


def summarize_discrepancy(
    times: np.ndarray,
    discrepancy: np.ndarray,
) -> DiscrepancySummary:
    """
    Summarize a discrepancy time series.

    Parameters
    ----------
    times:
        Time values.

    discrepancy:
        Corresponding discrepancy magnitude.

    Returns
    -------
    DiscrepancySummary
        Basic statistics describing the evolution of the discrepancy.
    """

    times = np.asarray(times, dtype=float)
    discrepancy = np.asarray(discrepancy, dtype=float)

    if times.ndim != 1 or discrepancy.ndim != 1:
        raise ValueError(
            "times and discrepancy must be one-dimensional."
        )

    if len(times) != len(discrepancy):
        raise ValueError(
            "times and discrepancy must have the same length."
        )

    if len(times) == 0:
        raise ValueError(
            "times and discrepancy cannot be empty."
        )

    maximum_index = int(
        np.argmax(discrepancy)
    )

    return DiscrepancySummary(
        initial=float(discrepancy[0]),
        final=float(discrepancy[-1]),
        maximum=float(np.max(discrepancy)),
        mean=float(np.mean(discrepancy)),
        time_of_maximum=float(times[maximum_index]),
    )


# ---------------------------------------------------------------------------
# Commutator-style diagnostic
# ---------------------------------------------------------------------------

def evolution_coarse_grain_discrepancy(
    coarse_from_micro: np.ndarray,
    effective_state: np.ndarray,
) -> np.ndarray:
    """
    Calculate the discrepancy between the two routes.

    This is the computational representation of

        Delta(t) =
            C[Phi_micro(t, X0)]
            - Phi_eff(t, C[X0])

    Parameters
    ----------
    coarse_from_micro:
        State obtained by evolving microscopically and then coarse-graining.

    effective_state:
        State obtained by coarse-graining initially and then evolving
        independently.

    Returns
    -------
    np.ndarray
        Component-wise Delta(t).
    """

    coarse_from_micro = np.asarray(
        coarse_from_micro,
        dtype=float,
    )

    effective_state = np.asarray(
        effective_state,
        dtype=float,
    )

    if coarse_from_micro.shape != effective_state.shape:
        raise ValueError(
            "The two states must have matching shapes."
        )

    return coarse_from_micro - effective_state


def commutation_error(
    coarse_from_micro: np.ndarray,
    effective_state: np.ndarray,
) -> float:
    """
    Return the RMS failure of the two operations to commute.

    Conceptually:

        C o Phi_micro
        ----------------
        versus
        Phi_eff o C

    A value near zero indicates approximate commutation for the supplied
    state and time.

    A larger value indicates that evolving and coarse-graining in the two
    different orders produce increasingly different representations.
    """

    delta = evolution_coarse_grain_discrepancy(
        coarse_from_micro,
        effective_state,
    )

    return rms(delta)
