"""
Tests for coarse-graining and basic representation metrics.

These tests are intentionally simple.

The goal at this stage is not to test the scientific hypothesis. It is to
verify that the computational machinery behaves exactly as specified.

In particular, we test:

1. Block averaging.
2. Shape validation.
3. Preservation of constant fields.
4. RMS and relative-error calculations.
5. State discrepancy calculations.
6. The commutation-error diagnostic.
"""

from __future__ import annotations

import numpy as np
import pytest

from scale_boundary.coarse_grain import (
    coarse_grain,
    coarse_grain_scalar,
)
from scale_boundary.metrics import (
    commutation_error,
    relative_error,
    rms,
    state_discrepancy,
)


# ---------------------------------------------------------------------------
# Coarse-graining tests
# ---------------------------------------------------------------------------

def test_coarse_grain_block_average():
    """
    Verify that block averaging produces the expected coarse values.
    """

    q = np.array([
        1.0,
        3.0,
        5.0,
        7.0,
    ])

    p = np.array([
        2.0,
        4.0,
        6.0,
        8.0,
    ])

    Q, P = coarse_grain(
        q,
        p,
        block_size=2,
    )

    expected_Q = np.array([
        2.0,
        6.0,
    ])

    expected_P = np.array([
        3.0,
        7.0,
    ])

    np.testing.assert_allclose(
        Q,
        expected_Q,
    )

    np.testing.assert_allclose(
        P,
        expected_P,
    )


def test_coarse_grain_constant_field():
    """
    A constant field should remain unchanged by block averaging.
    """

    q = np.full(12, 5.0)
    p = np.full(12, -2.0)

    Q, P = coarse_grain(
        q,
        p,
        block_size=3,
    )

    np.testing.assert_allclose(
        Q,
        5.0,
    )

    np.testing.assert_allclose(
        P,
        -2.0,
    )


def test_coarse_grain_scalar():
    """
    Verify coarse-graining of a single scalar field.
    """

    values = np.array([
        0.0,
        2.0,
        4.0,
        6.0,
    ])

    result = coarse_grain_scalar(
        values,
        block_size=2,
    )

    expected = np.array([
        1.0,
        5.0,
    ])

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_coarse_grain_rejects_incompatible_block_size():
    """
    A lattice that cannot be divided into equal blocks should fail.
    """

    q = np.arange(5, dtype=float)
    p = np.arange(5, dtype=float)

    with pytest.raises(ValueError):
        coarse_grain(
            q,
            p,
            block_size=2,
        )


def test_coarse_grain_rejects_mismatched_shapes():
    """
    q and p must describe the same number of microscopic sites.
    """

    q = np.zeros(8)
    p = np.zeros(4)

    with pytest.raises(ValueError):
        coarse_grain(
            q,
            p,
            block_size=2,
        )


def test_coarse_grain_rejects_invalid_block_size():
    """
    Block size must be positive.
    """

    q = np.arange(4, dtype=float)
    p = np.arange(4, dtype=float)

    with pytest.raises(ValueError):
        coarse_grain(
            q,
            p,
            block_size=0,
        )


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------

def test_rms_zero_for_zero_array():
    """The RMS of an all-zero array is zero."""

    values = np.zeros(10)

    assert rms(values) == pytest.approx(0.0)


def test_rms_known_value():
    """
    Check RMS against a simple analytical result.

    For [3, 4]:

        RMS = sqrt((9 + 16) / 2)
            = sqrt(12.5)
    """

    values = np.array([
        3.0,
        4.0,
    ])

    expected = np.sqrt(12.5)

    assert rms(values) == pytest.approx(expected)


def test_relative_error_zero_for_identical_arrays():
    """
    Identical reference and approximation states should have zero
    relative error.
    """

    values = np.array([
        1.0,
        2.0,
        3.0,
    ])

    assert relative_error(
        values,
        values,
    ) == pytest.approx(0.0)


def test_relative_error_known_difference():
    """
    Verify the relative-error calculation for a simple case.
    """

    reference = np.array([
        1.0,
        1.0,
    ])

    approximation = np.array([
        2.0,
        2.0,
    ])

    # RMS difference = 1
    # RMS reference = 1
    # Relative error = 1
    assert relative_error(
        reference,
        approximation,
    ) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# State discrepancy tests
# ---------------------------------------------------------------------------

def test_state_discrepancy_zero_for_identical_states():
    """
    Identical states should have zero discrepancy.
    """

    q = np.array([
        1.0,
        2.0,
        3.0,
    ])

    p = np.array([
        4.0,
        5.0,
        6.0,
    ])

    dq, dp, total = state_discrepancy(
        q,
        p,
        q,
        p,
    )

    assert dq == pytest.approx(0.0)
    assert dp == pytest.approx(0.0)
    assert total == pytest.approx(0.0)


def test_state_discrepancy_known_difference():
    """
    Verify the combined state discrepancy.

    q difference:
        [1, 1]

    p difference:
        [2, 2]

    RMS(q) = 1
    RMS(p) = 2

    Combined discrepancy:

        sqrt(1^2 + 2^2)
        = sqrt(5)
    """

    reference_q = np.array([
        1.0,
        1.0,
    ])

    effective_q = np.array([
        2.0,
        2.0,
    ])

    reference_p = np.array([
        1.0,
        1.0,
    ])

    effective_p = np.array([
        3.0,
        3.0,
    ])

    dq, dp, total = state_discrepancy(
        reference_q,
        reference_p,
        effective_q,
        effective_p,
    )

    assert dq == pytest.approx(1.0)
    assert dp == pytest.approx(2.0)
    assert total == pytest.approx(np.sqrt(5.0))


# ---------------------------------------------------------------------------
# Commutation tests
# ---------------------------------------------------------------------------

def test_commutation_error_zero_for_identical_states():
    """
    If both routes produce exactly the same state, the commutation error
    must be zero.
    """

    state = np.array([
        1.0,
        2.0,
        3.0,
    ])

    error = commutation_error(
        state,
        state,
    )

    assert error == pytest.approx(0.0)


def test_commutation_error_known_difference():
    """
    Check the RMS commutation error against a known difference.
    """

    coarse_from_micro = np.array([
        1.0,
        2.0,
        3.0,
    ])

    effective_state = np.array([
        2.0,
        4.0,
        6.0,
    ])

    expected = np.sqrt(
        (1.0 + 4.0 + 9.0) / 3.0
    )

    error = commutation_error(
        coarse_from_micro,
        effective_state,
    )

    assert error == pytest.approx(expected)


def test_commutation_error_rejects_mismatched_shapes():
    """
    The two routes must produce states in the same representation before
    they can be compared.
    """

    coarse_from_micro = np.zeros(4)
    effective_state = np.zeros(2)

    with pytest.raises(ValueError):
        commutation_error(
            coarse_from_micro,
            effective_state,
        )
