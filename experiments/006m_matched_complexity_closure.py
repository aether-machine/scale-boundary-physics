"""
Experiment 006m — Matched-Complexity Closure

Question:
    Does explicitly retaining the moments required by the exact moment
    hierarchy remove the apparent predictive advantage of coarse history?

Microscopic dynamics:
    dx/dt = x + x^3 - x^5

Exact moment hierarchy:
    dm_n/dt = n (m_n + m_{n+2} - m_{n+4})

Therefore:

    dm1/dt = m1 + m3 - m5
    dm2/dt = 2(m2 + m4 - m6)

This experiment compares higher-moment representations with finite-history
representations using the same microscopic system, independent training and
test trajectories, and the same ridge-regression framework.

Representations:

    m1_m2
    m1_to_m6
    m1_to_m8

    m1_m2_history1
    m1_m2_history2
    m1_m2_history4
    m1_m2_history8

The main question is not simply which representation has the lowest error.
The number of input features is reported so that model complexity can be
considered explicitly.

Outputs:

    results/006m_matched_complexity_closure.csv
    results/006m_matched_complexity_closure_summary.txt
    results/006m_matched_complexity_closure.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RNG_SEED = 6013

N_ENSEMBLES = 1600
N_TRAIN = 1200
N_TEST = 400

N_PARTICLES = 512

DT = 0.01
T_TOTAL = 8.0

RIDGE_ALPHA = 1e-4
N_BOOTSTRAP = 500

RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# Microscopic dynamics
# ---------------------------------------------------------------------------

def microscopic_rhs(x: np.ndarray) -> np.ndarray:
    """Return dx/dt."""
    return x + x**3 - x**5


def rk4_step(x: np.ndarray, dt: float) -> np.ndarray:
    """Advance the microscopic ensemble by one RK4 step."""
    k1 = microscopic_rhs(x)
    k2 = microscopic_rhs(x + 0.5 * dt * k1)
    k3 = microscopic_rhs(x + 0.5 * dt * k2)
    k4 = microscopic_rhs(x + dt * k3)

    return x + (dt / 6.0) * (
        k1 + 2.0 * k2 + 2.0 * k3 + k4
    )


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

def compute_moments(
    x: np.ndarray,
    max_order: int,
) -> np.ndarray:
    """Return m1 ... m_max_order."""
    return np.asarray(
        [np.mean(x**n) for n in range(1, max_order + 1)],
        dtype=float,
    )


def compute_target_derivative(
    x: np.ndarray,
) -> np.ndarray:
    """
    Exact derivatives of the retained target variables m1 and m2.

    dm1/dt = m1 + m3 - m5
    dm2/dt = 2(m2 + m4 - m6)
    """
    moments = compute_moments(x, 6)

    m1, m2, m3, m4, m5, m6 = moments

    dm1 = m1 + m3 - m5
    dm2 = 2.0 * (m2 + m4 - m6)

    return np.array([dm1, dm2], dtype=float)


# ---------------------------------------------------------------------------
# Initial ensembles
# ---------------------------------------------------------------------------

def make_initial_ensemble(
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a varied but bounded microscopic ensemble.

    A three-component mixture gives variation in higher moments while
    keeping the system in the same general dynamical regime as 006g-k.
    """
    component = rng.integers(
        0,
        3,
        size=N_PARTICLES,
    )

    means = np.array([
        0.70,
        1.00,
        1.30,
    ])

    stds = np.array([
        0.08,
        0.10,
        0.08,
    ])

    x = (
        means[component]
        + stds[component] * rng.normal(
            size=N_PARTICLES
        )
    )

    return x


# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------

def simulate_trajectory(
    x0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate one microscopic trajectory.

    Returns:

        moments:
            shape = (n_times, 8)

        targets:
            shape = (n_times, 2)
    """
    n_steps = int(round(T_TOTAL / DT))

    moments_history = []
    targets_history = []

    x = x0.copy()

    for step in range(n_steps + 1):
        moments_history.append(
            compute_moments(x, 8)
        )

        targets_history.append(
            compute_target_derivative(x)
        )

        x = rk4_step(x, DT)

    return (
        np.asarray(moments_history),
        np.asarray(targets_history),
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def generate_dataset(
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate independent microscopic trajectories."""
    all_moments = []
    all_targets = []

    print("Generating microscopic trajectories...")

    for i in range(N_ENSEMBLES):
        x0 = make_initial_ensemble(rng)

        moments, targets = simulate_trajectory(x0)

        all_moments.append(moments)
        all_targets.append(targets)

        if (i + 1) % 100 == 0:
            print(
                f"  generated {i + 1}/{N_ENSEMBLES}"
            )

    return all_moments, all_targets


# ---------------------------------------------------------------------------
# Supervised data
# ---------------------------------------------------------------------------

def assemble_data(
    trajectories_moments: list[np.ndarray],
    trajectories_targets: list[np.ndarray],
    moment_order: int,
    history_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build feature matrix and target matrix.

    Feature ordering:

        H_t,
        H_{t-1},
        ...
        H_{t-k}

    where H contains moments m1 ... m_moment_order.
    """
    X_parts = []
    Y_parts = []

    for moments, targets in zip(
        trajectories_moments,
        trajectories_targets,
    ):
        retained = moments[:, :moment_order]

        for t in range(
            history_length,
            len(retained),
        ):
            feature = []

            for lag in range(
                history_length + 1
            ):
                feature.extend(
                    retained[t - lag]
                )

            X_parts.append(feature)
            Y_parts.append(targets[t])

    return (
        np.asarray(X_parts),
        np.asarray(Y_parts),
    )


# ---------------------------------------------------------------------------
# Ridge regression
# ---------------------------------------------------------------------------

def standardize(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardize using training statistics only."""
    mean = np.mean(
        X_train,
        axis=0,
    )

    std = np.std(
        X_train,
        axis=0,
    )

    std = np.where(
        std < 1e-10,
        1.0,
        std,
    )

    return (
        (X_train - mean) / std,
        (X_test - mean) / std,
    )


def fit_ridge(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit multivariate ridge regression."""
    n_features = X.shape[1]

    A = X.T @ X
    A += alpha * np.eye(n_features)

    B = X.T @ Y

    return np.linalg.solve(A, B)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rms_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """RMS error over both target derivatives."""
    return float(
        np.sqrt(
            np.mean(
                (y_true - y_pred) ** 2
            )
        )
    )


def r_squared(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Multivariate R^2."""
    numerator = np.sum(
        (y_true - y_pred) ** 2
    )

    denominator = np.sum(
        (
            y_true
            - np.mean(
                y_true,
                axis=0,
            )
        ) ** 2
    )

    if denominator == 0:
        return float("nan")

    return float(
        1.0 - numerator / denominator
    )


def bootstrap_rms(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float]:
    """Bootstrap test RMS."""
    n = len(y_true)

    values = []

    for _ in range(n_bootstrap):
        indices = rng.integers(
            0,
            n,
            size=n,
        )

        values.append(
            rms_error(
                y_true[indices],
                y_pred[indices],
            )
        )

    values = np.asarray(values)

    return (
        float(np.median(values)),
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5)),
    )


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------

REPRESENTATIONS = [
    ("m1_m2", 2, 0),
    ("m1_to_m6", 6, 0),
    ("m1_to_m8", 8, 0),
    ("m1_m2_history1", 2, 1),
    ("m1_m2_history2", 2, 2),
    ("m1_m2_history4", 2, 4),
    ("m1_m2_history8", 2, 8),
]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_representation(
    name: str,
    moment_order: int,
    history_length: int,
    train_moments: list[np.ndarray],
    train_targets: list[np.ndarray],
    test_moments: list[np.ndarray],
    test_targets: list[np.ndarray],
    rng: np.random.Generator,
) -> dict:
    """Train and evaluate one representation."""
    X_train, Y_train = assemble_data(
        train_moments,
        train_targets,
        moment_order,
        history_length,
    )

    X_test, Y_test = assemble_data(
        test_moments,
        test_targets,
        moment_order,
        history_length,
    )

    X_train_scaled, X_test_scaled = standardize(
        X_train,
        X_test,
    )

    coefficients = fit_ridge(
        X_train_scaled,
        Y_train,
        RIDGE_ALPHA,
    )

    Y_pred = X_test_scaled @ coefficients

    rms = rms_error(
        Y_test,
        Y_pred,
    )

    r2 = r_squared(
        Y_test,
        Y_pred,
    )

    boot_median, boot_low, boot_high = (
        bootstrap_rms(
            Y_test,
            Y_pred,
            rng,
            N_BOOTSTRAP,
        )
    )

    # Number of regression coefficients.
    n_parameters = (
        X_train.shape[1]
        * Y_train.shape[1]
    )

    result = {
        "representation": name,
        "moment_order": moment_order,
        "history_length": history_length,
        "features": X_train.shape[1],
        "parameters": n_parameters,
        "test_rms": rms,
        "test_r2": r2,
        "bootstrap_median_rms": boot_median,
        "bootstrap_lower": boot_low,
        "bootstrap_upper": boot_high,
    }

    print(
        f"{name:22s} "
        f"features={X_train.shape[1]:3d} "
        f"parameters={n_parameters:3d} "
        f"RMS={rms:.8e} "
        f"R2={r2:.6f} "
        f"CI=[{boot_low:.8e}, "
        f"{boot_high:.8e}]"
    )

    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(
    results: list[dict],
    path: Path,
) -> None:
    """Write results CSV."""
    columns = [
        "representation",
        "moment_order",
        "history_length",
        "features",
        "parameters",
        "test_rms",
        "test_r2",
        "bootstrap_median_rms",
        "bootstrap_lower",
        "bootstrap_upper",
    ]

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            ",".join(columns)
            + "\n"
        )

        for result in results:
            f.write(
                ",".join(
                    str(result[column])
                    for column in columns
                )
                + "\n"
            )


def write_summary(
    results: list[dict],
    path: Path,
) -> None:
    """Write human-readable summary."""
    best = min(
        results,
        key=lambda r: r["test_rms"],
    )

    baseline = next(
        r
        for r in results
        if r["representation"] == "m1_m2"
    )

    lines = [
        "Experiment 006m — Matched-Complexity Closure",
        "",
        "Microscopic dynamics:",
        "    dx/dt = x + x^3 - x^5",
        "",
        "Exact hierarchy:",
        "    dm_n/dt = n(m_n + m_{n+2} - m_{n+4})",
        "",
        "Question:",
        "    Does retaining the moments required by the exact",
        "    hierarchy remove the predictive advantage of history?",
        "",
        "Results:",
        "",
    ]

    for r in results:
        improvement = (
            1.0
            - r["test_rms"]
            / baseline["test_rms"]
        )

        lines.append(
            f"{r['representation']:22s} "
            f"features={r['features']:3d} "
            f"parameters={r['parameters']:3d} "
            f"RMS={r['test_rms']:.8e} "
            f"R2={r['test_r2']:.6f} "
            f"improvement={improvement:.6f}"
        )

    lines.extend(
        [
            "",
            "Best representation by test RMS:",
            f"    {best['representation']}",
            f"    RMS = {best['test_rms']:.8e}",
            "",
            "Interpretation:",
            "    Lower prediction error does not by itself establish",
            "    that one representation is fundamentally preferable.",
            "    Feature count and parameter count must be considered.",
            "",
            "Important control:",
            "    m1 through m6 contain every moment appearing in the",
            "    exact instantaneous derivatives of m1 and m2.",
        ]
    )

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def make_plot(
    results: list[dict],
    path: Path,
) -> None:
    """Plot test RMS by representation."""
    labels = [
        r["representation"]
        for r in results
    ]

    values = [
        r["test_rms"]
        for r in results
    ]

    x = np.arange(
        len(results)
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        x,
        values,
    )

    plt.xticks(
        x,
        labels,
        rotation=35,
        ha="right",
    )

    plt.ylabel(
        "Out-of-sample RMS error"
    )

    plt.title(
        "006m — Matched-complexity closure"
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
    )

    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(
        RNG_SEED
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_moments, all_targets = (
        generate_dataset(rng)
    )

    train_moments = (
        all_moments[:N_TRAIN]
    )

    train_targets = (
        all_targets[:N_TRAIN]
    )

    test_moments = (
        all_moments[
            N_TRAIN:N_TRAIN + N_TEST
        ]
    )

    test_targets = (
        all_targets[
            N_TRAIN:N_TRAIN + N_TEST
        ]
    )

    print()
    print(
        "Evaluating representations..."
    )
    print()

    results = []

    for (
        name,
        moment_order,
        history_length,
    ) in REPRESENTATIONS:

        result = evaluate_representation(
            name,
            moment_order,
            history_length,
            train_moments,
            train_targets,
            test_moments,
            test_targets,
            rng,
        )

        results.append(result)

    csv_path = (
        RESULTS_DIR
        / "006m_matched_complexity_closure.csv"
    )

    summary_path = (
        RESULTS_DIR
        / "006m_matched_complexity_closure_summary.txt"
    )

    plot_path = (
        RESULTS_DIR
        / "006m_matched_complexity_closure.png"
    )

    write_csv(
        results,
        csv_path,
    )

    write_summary(
        results,
        summary_path,
    )

    make_plot(
        results,
        plot_path,
    )

    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
