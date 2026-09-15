"""
Experiment 006i: Data-Driven Memory Closure

Question:
    Can the history of retained variables recover information about
    discarded microscopic degrees of freedom?

Microscopic dynamics:
    dx/dt = x + x^3 - x^5

Retained variables:
    m1 = E[x]
    m2 = E[x^2]

We compare:

    1. Instantaneous closure:
           dH/dt = F(H(t))

    2. Finite-memory closure:
           dH/dt =
               A0 H(t)
             + A1 H(t-dt)
             + ...
             + Ak H(t-k*dt)

The coefficients are learned from microscopic trajectories.

IMPORTANT:
    This is a diagnostic model. It is not intended to claim that
    the fitted linear memory kernel is a physical law.

The main question is whether out-of-sample prediction improves when
history is supplied to the effective description.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(12345)

N_PARTICLES = 12000

DT = 0.002
T = 8.0

TRAIN_TRAJECTORIES = 24
TEST_TRAJECTORIES = 12

# Number of historical samples supplied to the memory model.
HISTORY_LENGTHS = [1, 2, 4, 8, 16]

RIDGE = 1e-6


# ---------------------------------------------------------------------
# Microscopic dynamics
# ---------------------------------------------------------------------

def microscopic_rhs(x):
    """
    dx/dt = x + x^3 - x^5
    """
    return x + x**3 - x**5


def rk4_step(x, dt):
    """
    Fourth-order Runge-Kutta step.
    """
    k1 = microscopic_rhs(x)
    k2 = microscopic_rhs(x + 0.5 * dt * k1)
    k3 = microscopic_rhs(x + 0.5 * dt * k2)
    k4 = microscopic_rhs(x + dt * k3)

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ---------------------------------------------------------------------
# Moment calculation
# ---------------------------------------------------------------------

def moments(x):
    """
    Return retained moments m1 and m2.
    """
    m1 = np.mean(x)
    m2 = np.mean(x**2)

    return np.array([m1, m2])


def moment_derivatives(x):
    """
    Exact derivatives of m1 and m2.

    dm1/dt = E[x + x^3 - x^5]

    dm2/dt = 2 E[x (x + x^3 - x^5)]
           = 2 E[x^2 + x^4 - x^6]
    """

    m1_dot = np.mean(x + x**3 - x**5)

    m2_dot = 2.0 * np.mean(
        x**2 + x**4 - x**6
    )

    return np.array([m1_dot, m2_dot])


# ---------------------------------------------------------------------
# Initial microscopic ensembles
# ---------------------------------------------------------------------

def generate_initial_state(kind):
    """
    Generate different microscopic distributions.

    The different distributions provide different hidden structures
    while remaining within the same general dynamical system.
    """

    if kind == 0:
        x = RNG.normal(1.00, 0.20, N_PARTICLES)

    elif kind == 1:
        components = RNG.choice(
            [0, 1, 2],
            size=N_PARTICLES,
            p=[0.25, 0.50, 0.25],
        )

        means = np.array([0.72, 1.00, 1.28])

        x = (
            means[components]
            + RNG.normal(0.0, 0.08, N_PARTICLES)
        )

    elif kind == 2:
        x = RNG.normal(0.90, 0.28, N_PARTICLES)

    elif kind == 3:
        x = RNG.uniform(
            0.45,
            1.55,
            N_PARTICLES,
        )

    else:
        # Mildly skewed distribution.
        z = RNG.normal(0.0, 1.0, N_PARTICLES)
        x = 1.0 + 0.18 * z + 0.04 * (z**2 - 1.0)

    return x


# ---------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------

def simulate_trajectory(kind):
    """
    Generate one microscopic trajectory.

    Returns
    -------
    history : ndarray
        Shape (n_times, 2), containing m1 and m2.
    derivatives : ndarray
        Shape (n_times, 2), containing exact dm1/dt and dm2/dt.
    """

    n_steps = int(round(T / DT)) + 1

    history = np.zeros((n_steps, 2))
    derivatives = np.zeros((n_steps, 2))

    x = generate_initial_state(kind)

    for i in range(n_steps):
        history[i] = moments(x)
        derivatives[i] = moment_derivatives(x)

        if i < n_steps - 1:
            x = rk4_step(x, DT)

    return history, derivatives


# ---------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------

def build_dataset(
    trajectories,
    history_length,
):
    """
    Construct supervised learning data.

    Input:
        current retained state plus optional history.

    Target:
        exact microscopic derivative of retained variables.
    """

    X = []
    Y = []

    for history, derivatives in trajectories:

        start = history_length

        for i in range(start, len(history)):
            features = []

            # Current state.
            features.extend(history[i])

            # Historical states.
            for lag in range(1, history_length + 1):
                features.extend(history[i - lag])

            X.append(features)
            Y.append(derivatives[i])

    return np.asarray(X), np.asarray(Y)


# ---------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------

def fit_scaling(X):
    mean = np.mean(X, axis=0)
    scale = np.std(X, axis=0)

    scale[scale < 1e-12] = 1.0

    return mean, scale


def apply_scaling(X, mean, scale):
    return (X - mean) / scale


# ---------------------------------------------------------------------
# Ridge regression
# ---------------------------------------------------------------------

def fit_ridge(X, Y, ridge=RIDGE):
    """
    Solve:

        min ||X A - Y||^2 + ridge ||A||^2
    """

    XtX = X.T @ X
    n_features = XtX.shape[0]

    regularisation = ridge * np.eye(n_features)

    return np.linalg.solve(
        XtX + regularisation,
        X.T @ Y,
    )


def predict(X, coefficients):
    return X @ coefficients


# ---------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------

def rms_error(reference, prediction):
    difference = reference - prediction
    return float(
        np.sqrt(np.mean(difference**2))
    )


def trajectory_error(reference, prediction):
    """
    Mean Euclidean error across the two retained variables.
    """

    difference = reference - prediction

    return float(
        np.mean(
            np.sqrt(
                np.sum(difference**2, axis=1)
            )
        )
    )


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():

    print("Experiment 006i: Data-Driven Memory Closure")
    print()
    print("Microscopic dynamics:")
    print("    dx/dt = x + x^3 - x^5")
    print()
    print("Retained variables:")
    print("    m1 = E[x]")
    print("    m2 = E[x^2]")
    print()
    print("Question:")
    print("    Does adding history improve prediction of retained")
    print("    dynamics when hidden microscopic information is discarded?")
    print()

    # -------------------------------------------------------------
    # Generate independent training and testing trajectories.
    # -------------------------------------------------------------

    print("Generating trajectories...")

    training = []

    for i in range(TRAIN_TRAJECTORIES):
        kind = i % 5
        training.append(
            simulate_trajectory(kind)
        )

    testing = []

    for i in range(TEST_TRAJECTORIES):
        # Use different random seeds/distributions by rotating the
        # initial ensemble types.
        kind = (i + 2) % 5
        testing.append(
            simulate_trajectory(kind)
        )

    print(
        f"Training trajectories: {len(training)}"
    )

    print(
        f"Testing trajectories:  {len(testing)}"
    )

    # -------------------------------------------------------------
    # Results table.
    # -------------------------------------------------------------

    results = []

    # -------------------------------------------------------------
    # Instantaneous closure.
    # -------------------------------------------------------------

    print()
    print("Fitting instantaneous closure...")

    X_train, Y_train = build_dataset(
        training,
        history_length=0,
    )

    X_test, Y_test = build_dataset(
        testing,
        history_length=0,
    )

    train_mean, train_scale = fit_scaling(X_train)

    X_train_scaled = apply_scaling(
        X_train,
        train_mean,
        train_scale,
    )

    X_test_scaled = apply_scaling(
        X_test,
        train_mean,
        train_scale,
    )

    instantaneous_coefficients = fit_ridge(
        X_train_scaled,
        Y_train,
    )

    instantaneous_prediction = predict(
        X_test_scaled,
        instantaneous_coefficients,
    )

    instantaneous_error = rms_error(
        Y_test,
        instantaneous_prediction,
    )

    print(
        f"Instantaneous test RMS error = "
        f"{instantaneous_error:.10e}"
    )

    results.append(
        (
            0,
            instantaneous_error,
        )
    )

    # -------------------------------------------------------------
    # Memory closures.
    # -------------------------------------------------------------

    memory_errors = {}

    for history_length in HISTORY_LENGTHS:

        print()
        print(
            f"Fitting memory closure: "
            f"{history_length} historical samples"
        )

        X_train, Y_train = build_dataset(
            training,
            history_length,
        )

        X_test, Y_test = build_dataset(
            testing,
            history_length,
        )

        train_mean, train_scale = fit_scaling(
            X_train
        )

        X_train_scaled = apply_scaling(
            X_train,
            train_mean,
            train_scale,
        )

        X_test_scaled = apply_scaling(
            X_test,
            train_mean,
            train_scale,
        )

        coefficients = fit_ridge(
            X_train_scaled,
            Y_train,
        )

        prediction = predict(
            X_test_scaled,
            coefficients,
        )

        error = rms_error(
            Y_test,
            prediction,
        )

        memory_errors[history_length] = error

        results.append(
            (
                history_length,
                error,
            )
        )

        print(
            f"Test RMS error = "
            f"{error:.10e}"
        )

    # -------------------------------------------------------------
    # Summary.
    # -------------------------------------------------------------

    best_history = min(
        memory_errors,
        key=memory_errors.get,
    )

    best_error = memory_errors[best_history]

    absolute_improvement = (
        instantaneous_error
        - best_error
    )

    fractional_improvement = (
        absolute_improvement
        / instantaneous_error
    )

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    print(
        f"Instantaneous error = "
        f"{instantaneous_error:.10e}"
    )

    print(
        f"Best memory length = "
        f"{best_history}"
    )

    print(
        f"Best memory error = "
        f"{best_error:.10e}"
    )

    print(
        f"Absolute improvement = "
        f"{absolute_improvement:.10e}"
    )

    print(
        f"Fractional improvement = "
        f"{fractional_improvement:.6f}"
    )

    # -------------------------------------------------------------
    # Save results.
    # -------------------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "006i_memory_closure_results.csv"
    )

    with open(csv_path, "w", encoding="utf-8") as f:

        f.write(
            "history_length,test_rms_error,"
            "improvement_vs_instantaneous\n"
        )

        for history_length, error in results:

            improvement = (
                instantaneous_error - error
            )

            f.write(
                f"{history_length},"
                f"{error:.12e},"
                f"{improvement:.12e}\n"
            )

    summary_path = (
        OUTPUT_DIR
        / "006i_memory_closure_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Experiment 006i: "
            "Data-Driven Memory Closure\n\n"
        )

        f.write(
            "Microscopic dynamics:\n"
        )

        f.write(
            "    dx/dt = x + x^3 - x^5\n\n"
        )

        f.write(
            "Retained variables:\n"
        )

        f.write(
            "    m1 = E[x]\n"
            "    m2 = E[x^2]\n\n"
        )

        f.write(
            "Instantaneous closure test RMS error:\n"
        )

        f.write(
            f"    {instantaneous_error:.12e}\n\n"
        )

        f.write(
            "Best memory history length:\n"
        )

        f.write(
            f"    {best_history}\n\n"
        )

        f.write(
            "Best memory test RMS error:\n"
        )

        f.write(
            f"    {best_error:.12e}\n\n"
        )

        f.write(
            "Absolute improvement:\n"
        )

        f.write(
            f"    {absolute_improvement:.12e}\n\n"
        )

        f.write(
            "Fractional improvement:\n"
        )

        f.write(
            f"    {fractional_improvement:.12e}\n"
        )

    # -------------------------------------------------------------
    # Plot.
    # -------------------------------------------------------------

    history_values = [
        r[0]
        for r in results
    ]

    error_values = [
        r[1]
        for r in results
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        history_values,
        error_values,
        marker="o",
    )

    plt.axhline(
        instantaneous_error,
        linestyle="--",
        label="Instantaneous closure",
    )

    plt.xlabel(
        "Number of historical samples"
    )

    plt.ylabel(
        "Test RMS derivative error"
    )

    plt.title(
        "006i: Prediction Error vs Memory Depth"
    )

    plt.legend()
    plt.grid(True)

    plot_path = (
        OUTPUT_DIR
        / "006i_memory_depth.png"
    )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    print()
    print("Saved:")
    print(f"    {csv_path}")
    print(f"    {summary_path}")
    print(f"    {plot_path}")

    print()
    print(
        "Interpretation:"
    )

    print(
        "A lower out-of-sample error with increasing "
        "history would indicate that temporal history "
        "contains information about discarded microscopic "
        "degrees of freedom."
    )

    print(
        "A failure to improve would indicate that the "
        "retained history, at least in this form, does "
        "not provide enough information to reconstruct "
        "the missing state."
    )

    print()
    print(
        "This experiment does not assume that memory is "
        "a fundamental physical mechanism. It tests "
        "whether memory is empirically useful as a "
        "representation of discarded information."
    )


if __name__ == "__main__":
    main()
