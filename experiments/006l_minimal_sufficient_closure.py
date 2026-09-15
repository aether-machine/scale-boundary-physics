"""
Experiment 006l — Minimal Sufficient Closure

Question:
    Is the predictive information supplied by coarse history comparable
    to the information supplied by retaining additional moments?

Microscopic dynamics:
    dx/dt = x + x^3 - x^5

Moment hierarchy:
    d/dt m_n = n (m_n + m_{n+2} - m_{n+4})

We compare several reduced representations:

    A: m1, m2
    B: m1, m2 + 4-step history
    C: m1, m2 + 16-step history
    D: m1, m2, m3
    E: m1, m2, m3, m4
    F: m1, m2, m3, m4, m5
    G: m1, m2, m3 + 4-step history

For each representation, a ridge regression model is trained on
independent trajectories and evaluated out of sample.

The target is the instantaneous coarse derivative:

    d/dt (m1, m2)

The main outputs are:
    - test RMS error
    - test R^2
    - bootstrap uncertainty of test RMS
    - comparison of predictive performance per retained dimension

This experiment is intended to determine whether history provides
information comparable to explicitly retaining higher moments.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RNG_SEED = 6012

N_ENSEMBLES = 1600
N_TRAIN = 1200
N_TEST = 400

N_PARTICLES = 512

DT = 0.01
T_TOTAL = 8.0

# We sample the moment trajectories at this interval.
SAMPLE_EVERY = 1

# Ridge regularisation.
RIDGE_ALPHA = 1e-4

# Bootstrap repetitions for test RMS uncertainty.
N_BOOTSTRAP = 500

RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# Microscopic dynamics
# ---------------------------------------------------------------------------

def microscopic_rhs(x: np.ndarray) -> np.ndarray:
    """Return dx/dt for the microscopic system."""
    return x + x**3 - x**5


def rk4_step(x: np.ndarray, dt: float) -> np.ndarray:
    """Advance the microscopic ensemble by one RK4 step."""
    k1 = microscopic_rhs(x)
    k2 = microscopic_rhs(x + 0.5 * dt * k1)
    k3 = microscopic_rhs(x + 0.5 * dt * k2)
    k4 = microscopic_rhs(x + dt * k3)

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

def compute_moments(x: np.ndarray, max_order: int) -> np.ndarray:
    """Return moments m1 ... m_max_order."""
    return np.array(
        [np.mean(x**n) for n in range(1, max_order + 1)],
        dtype=float,
    )


def compute_moment_derivatives(x: np.ndarray) -> np.ndarray:
    """
    Exact microscopic derivatives of m1 and m2.

    dm_n/dt = n E[x^(n-1) dx/dt]

    For dx/dt = x + x^3 - x^5:

        dm_n/dt = n (m_n + m_{n+2} - m_{n+4})
    """
    moments = {
        n: np.mean(x**n)
        for n in range(1, 7)
    }

    dm1 = moments[1] + moments[3] - moments[5]
    dm2 = 2.0 * (
        moments[2] + moments[4] - moments[6]
    )

    return np.array([dm1, dm2], dtype=float)


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def make_initial_ensemble(
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a bounded ensemble with deliberately varied higher moments.

    Mixtures provide a range of hidden states while keeping the retained
    variables in a useful common region.
    """
    component = rng.integers(0, 3, size=N_PARTICLES)

    means = np.array([0.70, 1.00, 1.30])
    stds = np.array([0.08, 0.10, 0.08])

    x = (
        means[component]
        + stds[component] * rng.normal(size=N_PARTICLES)
    )

    return x


def simulate_trajectory(
    x0: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate one microscopic trajectory.

    Returns:
        moments: shape (n_times, 5)
        targets: shape (n_times, 2)
    """
    n_steps = int(round(T_TOTAL / DT))

    sampled_moments = []
    sampled_targets = []

    x = x0.copy()

    for step in range(n_steps + 1):
        if step % SAMPLE_EVERY == 0:
            sampled_moments.append(
                compute_moments(x, max_order=5)
            )
            sampled_targets.append(
                compute_moment_derivatives(x)
            )

        x = rk4_step(x, DT)

    return (
        np.asarray(sampled_moments),
        np.asarray(sampled_targets),
    )


# ---------------------------------------------------------------------------
# History construction
# ---------------------------------------------------------------------------

def build_features(
    moments: np.ndarray,
    history_length: int,
    moment_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build supervised learning features.

    The current retained moments are followed by past retained moments.

    history_length=0:
        [H_t]

    history_length=4:
        [H_t, H_{t-1}, ..., H_{t-4}]
    """
    retained = moments[:, :moment_order]

    features = []
    targets = []

    for t in range(history_length, len(retained)):
        history = []

        for lag in range(history_length + 1):
            history.extend(retained[t - lag])

        features.append(history)

        # Target is always the exact derivative of m1,m2.
        # It is supplied separately by the caller.
        targets.append(t)

    return np.asarray(features), np.asarray(targets)


# ---------------------------------------------------------------------------
# Ridge regression
# ---------------------------------------------------------------------------

def standardize_train_test(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features using training statistics only."""
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)

    std = np.where(std < 1e-10, 1.0, std)

    return (
        (X_train - mean) / std,
        (X_test - mean) / std,
        mean,
        std,
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


def predict_ridge(
    X: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Predict target derivatives."""
    return X @ coefficients


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rms_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean square error across both target dimensions."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Multivariate coefficient of determination."""
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum(
        (y_true - np.mean(y_true, axis=0)) ** 2
    )

    if denominator == 0:
        return float("nan")

    return float(1.0 - numerator / denominator)


def bootstrap_rms(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float, float]:
    """
    Bootstrap the RMS error by resampling test trajectories.

    Returns:
        median, lower 2.5%, upper 97.5%
    """
    n = len(y_true)

    values = []

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)

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
# Dataset assembly
# ---------------------------------------------------------------------------

def generate_dataset(
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Generate independent trajectory datasets."""
    all_moments = []
    all_targets = []

    print("Generating microscopic trajectories...")

    for i in range(N_ENSEMBLES):
        x0 = make_initial_ensemble(rng)

        moments, targets = simulate_trajectory(x0)

        all_moments.append(moments)
        all_targets.append(targets)

        if (i + 1) % 100 == 0:
            print(f"  generated {i + 1}/{N_ENSEMBLES}")

    return all_moments, all_targets


# ---------------------------------------------------------------------------
# Representation evaluation
# ---------------------------------------------------------------------------

REPRESENTATIONS = [
    ("m1_m2", 2, 0),
    ("m1_m2_history4", 2, 4),
    ("m1_m2_history16", 2, 16),
    ("m1_m2_m3", 3, 0),
    ("m1_to_m4", 4, 0),
    ("m1_to_m5", 5, 0),
    ("m1_m2_m3_history4", 3, 4),
]


def assemble_supervised_data(
    trajectories_moments: list[np.ndarray],
    trajectories_targets: list[np.ndarray],
    moment_order: int,
    history_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct features and targets for one representation."""
    X_parts = []
    Y_parts = []

    for moments, targets in zip(
        trajectories_moments,
        trajectories_targets,
    ):
        retained = moments[:, :moment_order]

        for t in range(history_length, len(retained)):
            feature = []

            for lag in range(history_length + 1):
                feature.extend(retained[t - lag])

            X_parts.append(feature)
            Y_parts.append(targets[t])

    return (
        np.asarray(X_parts),
        np.asarray(Y_parts),
    )


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
    """Train and evaluate one reduced representation."""
    X_train, Y_train = assemble_supervised_data(
        train_moments,
        train_targets,
        moment_order,
        history_length,
    )

    X_test, Y_test = assemble_supervised_data(
        test_moments,
        test_targets,
        moment_order,
        history_length,
    )

    X_train_scaled, X_test_scaled, _, _ = (
        standardize_train_test(
            X_train,
            X_test,
        )
    )

    coefficients = fit_ridge(
        X_train_scaled,
        Y_train,
        RIDGE_ALPHA,
    )

    Y_pred = predict_ridge(
        X_test_scaled,
        coefficients,
    )

    rms = rms_error(Y_test, Y_pred)
    r2 = r_squared(Y_test, Y_pred)

    median_boot, lower_boot, upper_boot = bootstrap_rms(
        Y_test,
        Y_pred,
        rng,
        N_BOOTSTRAP,
    )

    n_features = X_train.shape[1]

    result = {
        "representation": name,
        "moment_order": moment_order,
        "history_length": history_length,
        "features": n_features,
        "test_rms": rms,
        "test_r2": r2,
        "bootstrap_median_rms": median_boot,
        "bootstrap_lower": lower_boot,
        "bootstrap_upper": upper_boot,
    }

    print(
        f"{name:24s} "
        f"features={n_features:3d} "
        f"RMS={rms:.8e} "
        f"R2={r2:.6f} "
        f"CI=[{lower_boot:.8e}, {upper_boot:.8e}]"
    )

    return result


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_csv(results: list[dict], path: Path) -> None:
    """Write results as CSV without requiring pandas."""
    columns = [
        "representation",
        "moment_order",
        "history_length",
        "features",
        "test_rms",
        "test_r2",
        "bootstrap_median_rms",
        "bootstrap_lower",
        "bootstrap_upper",
    ]

    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(columns) + "\n")

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
    """Write a human-readable summary."""
    best = min(
        results,
        key=lambda r: r["test_rms"],
    )

    lines = [
        "Experiment 006l — Minimal Sufficient Closure",
        "",
        "Microscopic dynamics:",
        "    dx/dt = x + x^3 - x^5",
        "",
        "Question:",
        "    Does adding higher moments provide comparable predictive",
        "    information to adding coarse history?",
        "",
        "Results:",
        "",
    ]

    for r in results:
        lines.append(
            f"{r['representation']:24s} "
            f"features={r['features']:3d} "
            f"RMS={r['test_rms']:.8e} "
            f"R2={r['test_r2']:.6f} "
            f"bootstrap=[{r['bootstrap_lower']:.8e}, "
            f"{r['bootstrap_upper']:.8e}]"
        )

    lines.extend(
        [
            "",
            "Best representation by test RMS:",
            f"    {best['representation']}",
            f"    RMS = {best['test_rms']:.8e}",
            "",
            "Interpretation must account for representation dimension.",
            "A lower error alone does not establish that one form of",
            "information is fundamentally preferable to another.",
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

    x = np.arange(len(labels))

    plt.figure(figsize=(11, 6))
    plt.bar(x, values)

    plt.xticks(
        x,
        labels,
        rotation=35,
        ha="right",
    )

    plt.ylabel("Out-of-sample RMS error")
    plt.title(
        "006l — Minimal sufficient closure"
    )

    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_moments, all_targets = generate_dataset(rng)

    # Independent trajectory split.
    train_moments = all_moments[:N_TRAIN]
    train_targets = all_targets[:N_TRAIN]

    test_moments = all_moments[N_TRAIN:N_TRAIN + N_TEST]
    test_targets = all_targets[N_TRAIN:N_TRAIN + N_TEST]

    print()
    print("Evaluating representations...")
    print()

    results = []

    for name, moment_order, history_length in REPRESENTATIONS:
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

    csv_path = RESULTS_DIR / "006l_minimal_sufficient_closure.csv"
    summary_path = (
        RESULTS_DIR
        / "006l_minimal_sufficient_closure_summary.txt"
    )
    plot_path = (
        RESULTS_DIR
        / "006l_minimal_sufficient_closure.png"
    )

    write_csv(results, csv_path)
    write_summary(results, summary_path)
    make_plot(results, plot_path)

    print()
    print(f"Saved: {csv_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
