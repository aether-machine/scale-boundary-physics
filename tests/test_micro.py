"""Tests for the microscopic dynamical model."""

import numpy as np

from scale_boundary.micro import acceleration, rhs


def test_acceleration_zero_state():
    """A system at q=0 should have zero acceleration."""

    q = np.zeros(8)

    result = acceleration(
        q,
        k=1.0,
        omega0_sq=0.5,
        beta=0.25,
    )

    np.testing.assert_allclose(result, 0.0)


def test_acceleration_constant_state():
    """
    For a spatially constant displacement, the coupling term vanishes.

    The remaining restoring force is

        a = -omega0^2 q - beta q^3.
    """

    q = np.full(8, 2.0)

    expected = -0.5 * 2.0 - 0.25 * (2.0 ** 3)

    result = acceleration(
        q,
        k=1.0,
        omega0_sq=0.5,
        beta=0.25,
    )

    np.testing.assert_allclose(result, expected)


def test_acceleration_is_periodic():
    """
    The spatial coupling uses periodic boundary conditions.
    """

    q = np.zeros(4)
    q[0] = 1.0

    result = acceleration(
        q,
        k=1.0,
        omega0_sq=0.0,
        beta=0.0,
    )

    expected = np.array([
        -2.0,
        1.0,
        0.0,
        1.0,
    ])

    np.testing.assert_allclose(result, expected)


def test_acceleration_is_odd_in_displacement():
    """
    The model should satisfy

        a(-q) = -a(q)
    """

    q = np.array([
        0.2,
        -0.5,
        1.1,
        -0.7,
    ])

    a_positive = acceleration(q)
    a_negative = acceleration(-q)

    np.testing.assert_allclose(
        a_negative,
        -a_positive,
    )


def test_rhs_returns_velocity_and_acceleration():
    """The first-order system should return dq/dt=p and dp/dt=a(q)."""

    q = np.array([
        0.2,
        -0.5,
        1.1,
        -0.7,
    ])

    p = np.array([
        0.3,
        0.1,
        -0.2,
        0.4,
    ])

    dqdt, dpdt = rhs(q, p)

    np.testing.assert_allclose(dqdt, p)
    np.testing.assert_allclose(dpdt, acceleration(q))


def test_rhs_preserves_state_shape():
    """The RHS should preserve the shape of the state vectors."""

    q = np.zeros(16)
    p = np.ones(16)

    dqdt, dpdt = rhs(q, p)

    assert dqdt.shape == q.shape
    assert dpdt.shape == p.shape
