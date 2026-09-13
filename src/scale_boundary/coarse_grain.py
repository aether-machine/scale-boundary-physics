"""
Coarse-graining and representation maps.

The central operation in this module is the transformation

    C : microscopic state -> coarse state

The first implementation uses block averaging.

This deliberately discards intra-block information. That discarded
information is not assumed to be physically meaningless; measuring the
consequences of discarding it is one of the central purposes of the
project.
"""

from __future__ import annotations

import numpy as np


def validate_block_size(
    n: int,
    block_size: int,
) -> None:
    """Validate that a lattice can be divided into equal blocks."""

    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    if n % block_size != 0:
        raise ValueError(
            "Number of microscopic sites must divide evenly "
            "by block_size."
        )


def coarse_grain(
    q: np.ndarray,
    p: np.ndarray,
    block_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert microscopic oscillator variables into block averages.

    Parameters
    ----------
    q:
        Microscopic displacement field.
    p:
        Microscopic momentum field.
    block_size:
        Number of microscopic sites represented by each coarse site.

    Returns
    -------
    Q, P:
        Coarse displacement and momentum variables.
    """

    if q.shape != p.shape:
        raise ValueError("q and p must have the same shape.")

    validate_block_size(
        len(q),
        block_size,
    )

    n_blocks = len(q) // block_size

    q_blocks = q.reshape(
        n_blocks,
        block_size,
    )

    p_blocks = p.reshape(
        n_blocks,
        block_size,
    )

    Q = q_blocks.mean(axis=1)
    P = p_blocks.mean(axis=1)

    return Q, P


def coarse_grain_scalar(
    values: np.ndarray,
    block_size: int,
) -> np.ndarray:
    """
    Coarse-grain a single scalar field by block averaging.

    This utility is useful for later experiments involving quantities
    such as density, temperature, energy, or other observables.
    """

    validate_block_size(
        len(values),
        block_size,
    )

    n_blocks = len(values) // block_size

    return values.reshape(
        n_blocks,
        block_size,
    ).mean(axis=1)
