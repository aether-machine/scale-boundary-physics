"""Tests for the effective dynamical model."""

import numpy as np

from scale_boundary.effective import acceleration, rhs, rk4_step


def test_acceleration_zero_state():
    """An effective state at Q=0 should have zero acceleration."""

    Q = np.zeros(8)

    result = acceleration(
        Q,
        k=1.0,
        omega0_sq=0.5,
        beta=0.25,
    )

    np.testing.assert_allclose(result, 0.0)


def test_acceleration_constant_state():
    """
    For a spatially constant displacement, the coupling term vanishes.
    """

    Q = np.full(8, 2.0)

    expected = -0.5 * 2.0 - 0.25 * (2.0 ** 3)

    result = acceleration(
        Q,
        k=1.0,
        omega0_sq=0.5,
        beta=0.25,
    )

    np.testing.assert_allclose(result, expected)


def test_rhs_returns_velocity_and_acceleration():
    """The RHS should return dQ/dt=P and dP/dt=a(Q)."""

    Q = np.array([
        0.2,
        -0.5,
        1.1,
        -0.7,
    ])

    P = np.array([
        0.3,
        0.1,
        -0.2,
        0.4,
    ])

    dQdt, dPdt = rhs(Q, P)

    np.testing.assert_allclose(dQdt, P)
    np.testing.assert_allclose(dPdt, acceleration(Q))


def test_rhs_preserves_state_shape():
    """The RHS should preserve the shape of the state vectors."""

    Q = np.zeros(16)
    P = np.ones(16)

    dQdt, dPdt = rhs(Q, P)

    assert dQdt.shape == Q.shape
    assert dPdt.shape == P.shape


def test_rk4_zero_state_remains_zero():
    """RK4 should leave the zero state unchanged."""

    Q = np.zeros(8)
    P = np.zeros(8)

    Q_new, P_new = rk4_step(
        Q,
        P,
        dt=0.01,
    )

    np.testing.assert_allclose(Q_new, 0.0)
    np.testing.assert_allclose(P_new, 0.0)


def test_rk4_is_consistent_for_small_step():
    """
    A sufficiently small RK4 step should produce a small change
    from the initial state.
    """

    Q = np.array([
        0.1,
        -0.2,
        0.3,
        -0.4,
    ])

    P = np.array([
        0.05,
        -0.1,
        0.15,
        -0.2,
    ])

    dt = 1e-5

    Q_new, P_new = rk4_step(
        Q,
        P,
        dt=dt,
    )

    assert np.all(np.isfinite(Q_new))
    assert np.all(np.isfinite(P_new))

    np.testing.assert_allclose(
        Q_new,
        Q + dt * P,
        rtol=1e-8,
        atol=1e-10,
    )
