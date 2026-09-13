"""Tests for the effective dynamical model."""

import numpy as np

from scale_boundary.effective import acceleration, rhs, rk4_step

def test_acceleration_zero_state():
"""An effective state at Q=0 should have zero acceleration."""

```
q = np.zeros(8)

result = acceleration(
    q,
    k=1.0,
    omega0_sq=0.5,
    beta=0.25,
)

np.testing.assert_allclose(result, 0.0)
```

def test_acceleration_constant_state():
"""
For a spatially constant displacement, the coupling term vanishes.

```
The remaining restoring force is

    a = -omega0^2 Q - beta Q^3.
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
```

def test_acceleration_is_periodic():
"""
The effective lattice uses periodic boundary conditions.

```
A displacement at the first site interacts with both neighboring
sites through the periodic lattice.
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
```

def test_acceleration_is_odd_in_displacement():
"""
The effective model should satisfy

```
    a(-Q) = -a(Q).
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
```

def test_rhs_returns_velocity_and_acceleration():
"""The first-order effective system should return dQ/dt=P."""

```
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
```

def test_rhs_preserves_state_shape():
"""The effective RHS should preserve state-vector shape."""

```
q = np.zeros(16)
p = np.ones(16)

dqdt, dpdt = rhs(q, p)

assert dqdt.shape == q.shape
assert dpdt.shape == p.shape
```

def test_rk4_zero_state_remains_zero():
"""A zero state should remain unchanged under RK4 integration."""

```
q = np.zeros(8)
p = np.zeros(8)

q_next, p_next = rk4_step(
    q,
    p,
    dt=0.01,
    k=1.0,
    omega0_sq=0.5,
    beta=0.25,
)

np.testing.assert_allclose(q_next, q)
np.testing.assert_allclose(p_next, p)
```

def test_rk4_preserves_state_shape():
"""One RK4 step should preserve the shape of both state vectors."""

```
q = np.linspace(-0.5, 0.5, 16)
p = np.linspace(0.5, -0.5, 16)

q_next, p_next = rk4_step(
    q,
    p,
    dt=0.001,
    k=1.0,
    omega0_sq=0.5,
    beta=0.25,
)

assert q_next.shape == q.shape
assert p_next.shape == p.shape
```

def test_rk4_is_consistent_for_small_step():
"""
For sufficiently small dt,

```
    Q(t + dt) = Q(t) + dt * P(t) + O(dt^2).

RK4 should therefore agree closely with this first-order
prediction for a sufficiently small timestep.
"""

q = np.array([
    0.2,
    -0.1,
    0.3,
    -0.4,
])

p = np.array([
    0.5,
    -0.2,
    0.1,
    0.3,
])

dt = 1e-5

q_next, p_next = rk4_step(
    q,
    p,
    dt=dt,
    k=1.0,
    omega0_sq=0.5,
    beta=0.25,
)

expected_q = q + dt * p

np.testing.assert_allclose(
    q_next,
    expected_q,
    rtol=1e-8,
    atol=1e-10,
)
```
