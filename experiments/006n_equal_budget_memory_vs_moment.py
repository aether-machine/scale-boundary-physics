"""
Experiment 006n — Equal-Budget Memory vs Moment Closure

Question:
    At comparable model capacity, does temporal history provide more
    predictive information than additional instantaneous moments?

Microscopic dynamics:
    dx/dt = x + x^3 - x^5

Exact moment hierarchy:
    dm_n/dt = n(m_n + m_{n+2} - m_{n+4})

Therefore:

    dm1/dt = m1 + m3 - m5
    dm2/dt = 2(m2 + m4 - m6)

The experiment compares moment-only and history-based representations
under approximately equal regression parameter budgets.

For two target variables, the number of regression coefficients is:

    parameters = features * 2

We therefore construct matched pairs:

    Budget ~ 8:
        m1_m2_history1       = 4 features / 8 parameters
        m1_to_m4             = 4 features / 8 parameters

    Budget ~ 12:
        m1_m2_history2       = 6 features / 12 parameters
        m1_to_m6             = 6 features / 12 parameters

    Budget ~ 16:
        m1_m2_history3       = 8 features / 16 parameters
        m1_to_m8             = 8 features / 16 parameters

    Budget ~ 20:
        m1_m2_history4       = 10 features / 20 parameters
        m1_to_m10            = 10 features / 20 parameters

    Budget ~ 28:
        m1_m2_history6       = 14 features / 28 parameters
        m1_to_m14            = 14 features / 28 parameters

    Budget ~ 36:
        m1_m2_history8       = 18 features / 36 parameters
        m1_to_m18            = 18 features / 36 parameters

The main comparison is between representations with exactly equal
numbers of features and regression coefficients.

Multiple independent train/test splits are used so that a single
favourable split cannot determine the result.

Outputs:

    results/006n_equal_budget_memory_vs_moment.csv
    results/006n_equal_budget_memory_vs_moment_summary.txt
    results/006n_equal_budget_memory_vs_moment.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_SEED = 6101

N_ENSEMBLES = 1000
N_TRAIN = 700
N_TEST = 300

N_PARTICLES = 512

DT = 0.01
T_TOTAL = 8.0

RIDGE_ALPHA = 1e-4

N_SPLITS = 8
N_BOOTSTRAP = 300

RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# Microscopic dynamics
# ---------------------------------------------------------------------------

def microscopic_rhs(x: np.ndarray) -> np.ndarray:
    """dx/dt = x + x^3 - x^5."""
    return x + x**3 - x**5


def rk4_step(
    x: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Advance the microscopic ensemble by one RK4 step."""
    k1 = microscopic_rhs(x)
    k2 = microscopic_rhs(x + 0.5 * dt * k1)
    k3 = microscopic_rhs(x + 0.5 * dt * k2)
    k4 = microscopic_rhs(x + dt * k3)

    return x + (
        dt / 6.0
    ) * (
        k1
        + 2.0 * k2
        + 2.0 * k3
        + k4
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
        [
            np.mean(x**n)
            for n in range(1, max_order + 1)
        ],
        dtype=float,
    )


def compute_target_derivative(
    x: np.ndarray,
) -> np.ndarray:
    """
    Exact derivatives of m1 and m2.

    dm1/dt = m1 + m3 - m5
    dm2/dt = 2(m2 + m4 - m6)
    """
    moments = compute_moments(
        x,
        6,
    )

    m1, m2, m3, m4, m5, m6 = moments

    dm1 = m1 + m3 - m5
    dm2 = 2.0 * (
        m2 + m4 - m6
    )

    return np.array(
        [dm1, dm2],
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Initial ensembles
# ---------------------------------------------------------------------------

def make_initial_ensemble(
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate a varied bounded microscopic ensemble.

    Three nearby components provide variation in higher moments while
    keeping trajectories in the same general dynamical regime used
    throughout the 006 series.
    """
    component = rng.integers(
        0,
        3,
        size=N_PARTICLES,
    )

    means = np.array(
        [0.70, 1.00, 1.30]
    )

    stds = np.array(
        [0.08, 0.10, 0.08]
    )

    return (
        means[component]
        + stds[component]
        * rng.normal(
            size=N_PARTICLES
        )
    )


# ---------------------------------------------------------------------------
# Trajectory generation
# ---------------------------------------------------------------------------

def simulate_trajectory(
    x0: np.ndarray,
    max_moment_order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate one trajectory.

    Returns:

        moments:
            shape = (n_times, max_moment_order)

        targets:
            shape = (n_times, 2)
    """
    n_steps = int(
        round(T_TOTAL / DT)
    )

    moments_history = []
    targets_history = []

    x = x0.copy()

    for _ in range(n_steps + 1):
        moments_history.append(
            compute_moments(
                x,
                max_moment_order,
            )
        )

        targets_history.append(
            compute_target_derivative(x)
        )

        x = rk4_step(
            x,
            DT,
        )

    return (
        np.asarray(
            moments_history
        ),
        np.asarray(
            targets_history
        ),
    )


def generate_dataset(
    rng: np.random.Generator,
    max_moment_order: int,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
]:
    """Generate all independent trajectories."""
    moments = []
    targets = []

    print(
        "Generating microscopic trajectories..."
    )

    for i in range(N_ENSEMBLES):
        x0 = make_initial_ensemble(
            rng
        )

        trajectory_moments, trajectory_targets = (
            simulate_trajectory(
                x0,
                max_moment_order,
            )
        )

        moments.append(
            trajectory_moments
        )

        targets.append(
            trajectory_targets
        )

        if (i + 1) % 100 == 0:
            print(
                f"  generated "
                f"{i + 1}/{N_ENSEMBLES}"
            )

    return moments, targets


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def assemble_data(
    trajectories_moments: list[np.ndarray],
    trajectories_targets: list[np.ndarray],
    moment_order: int,
    history_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct supervised learning data.

    Features are:

        H_t,
        H_{t-1},
        ...
        H_{t-k}

    where H contains m1 ... m_moment_order.
    """
    X = []
    Y = []

    for moments, targets in zip(
        trajectories_moments,
        trajectories_targets,
    ):
        retained = moments[
            :,
            :moment_order,
        ]

        for t in range(
            history_length,
            len(retained),
        ):
            features = []

            for lag in range(
                history_length + 1
            ):
                features.extend(
                    retained[t - lag]
                )

            X.append(features)
            Y.append(targets[t])

    return (
        np.asarray(X),
        np.asarray(Y),
    )


# ---------------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------------

def standardize(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Standardise using training statistics only."""
    mean = np.mean(
        X_train,
        axis=0,
    )

    std = np.std(
        X_train,
        axis=0,
    )

    std = np.where(
        std < 1e-12,
        1.0,
        std,
    )

    return (
        (X_train - mean) / std,
        (X_test - mean) / std,
    )


# ---------------------------------------------------------------------------
# Ridge regression
# ---------------------------------------------------------------------------

def fit_ridge(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit multivariate ridge regression."""
    n_features = X.shape[1]

    A = (
        X.T @ X
        + alpha * np.eye(
            n_features
        )
    )

    B = X.T @ Y

    return np.linalg.solve(
        A,
        B,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rms_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """RMS error across both targets."""
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
    """Multivariate R²."""
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

    return float(
        1.0
        - numerator / denominator
    )


def bootstrap_rms(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for RMS."""
    n = len(y_true)

    values = []

    for _ in range(
        N_BOOTSTRAP
    ):
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
        float(
            np.median(values)
        ),
        float(
            np.percentile(
                values,
                2.5,
            )
        ),
        float(
            np.percentile(
                values,
                97.5,
            )
        ),
    )


# ---------------------------------------------------------------------------
# Representation definitions
# ---------------------------------------------------------------------------

MATCHED_PAIRS = [
    {
        "budget": 8,
        "moment_order": 4,
        "history_length": 1,
        "moment_name": "m1_to_m4",
        "history_name": "m1_m2_history1",
    },
    {
        "budget": 12,
        "moment_order": 6,
        "history_length": 2,
        "moment_name": "m1_to_m6",
        "history_name": "m1_m2_history2",
    },
    {
        "budget": 16,
        "moment_order": 8,
        "history_length": 3,
        "moment_name": "m1_to_m8",
        "history_name": "m1_m2_history3",
    },
    {
        "budget": 20,
        "moment_order": 10,
        "history_length": 4,
        "moment_name": "m1_to_m10",
        "history_name": "m1_m2_history4",
    },
    {
        "budget": 28,
        "moment_order": 14,
        "history_length": 6,
        "moment_name": "m1_to_m14",
        "history_name": "m1_m2_history6",
    },
    {
        "budget": 36,
        "moment_order": 18,
        "history_length": 8,
        "moment_name": "m1_to_m18",
        "history_name": "m1_m2_history8",
    },
]


# ---------------------------------------------------------------------------
# One representation / split
# ---------------------------------------------------------------------------

def evaluate_representation(
    name: str,
    moment_order: int,
    history_length: int,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    all_moments: list[np.ndarray],
    all_targets: list[np.ndarray],
    rng: np.random.Generator,
) -> dict:
    """Train and evaluate one representation."""
    train_moments = [
        all_moments[i]
        for i in train_indices
    ]

    train_targets = [
        all_targets[i]
        for i in train_indices
    ]

    test_moments = [
        all_moments[i]
        for i in test_indices
    ]

    test_targets = [
        all_targets[i]
        for i in test_indices
    ]

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

    X_train, X_test = standardize(
        X_train,
        X_test,
    )

    coefficients = fit_ridge(
        X_train,
        Y_train,
        RIDGE_ALPHA,
    )

    prediction = (
        X_test @ coefficients
    )

    rms = rms_error(
        Y_test,
        prediction,
    )

    r2 = r_squared(
        Y_test,
        prediction,
    )

    boot_median, boot_low, boot_high = (
        bootstrap_rms(
            Y_test,
            prediction,
            rng,
        )
    )

    n_features = X_train.shape[1]
    n_parameters = (
        n_features
        * Y_train.shape[1]
    )

    return {
        "representation": name,
        "moment_order": moment_order,
        "history_length": history_length,
        "features": n_features,
        "parameters": n_parameters,
        "test_rms": rms,
        "test_r2": r2,
        "bootstrap_median_rms": boot_median,
        "bootstrap_lower": boot_low,
        "bootstrap_upper": boot_high,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    master_rng = np.random.default_rng(
        BASE_SEED
    )

    # The highest moment representation is m1...m18.
    all_moments, all_targets = (
        generate_dataset(
            master_rng,
            max_moment_order=18,
        )
    )

    n = len(all_moments)

    all_split_results = []

    print()
    print(
        "Running matched-budget splits..."
    )
    print()

    for split in range(
        N_SPLITS
    ):
        split_rng = np.random.default_rng(
            BASE_SEED + 1000 + split
        )

        indices = np.arange(n)

        split_rng.shuffle(
            indices
        )

        train_indices = indices[
            :N_TRAIN
        ]

        test_indices = indices[
            N_TRAIN:
            N_TRAIN + N_TEST
        ]

        print(
            f"Split {split + 1}/{N_SPLITS}"
        )

        for pair in MATCHED_PAIRS:

            moment_result = (
                evaluate_representation(
                    pair["moment_name"],
                    pair["moment_order"],
                    0,
                    train_indices,
                    test_indices,
                    all_moments,
                    all_targets,
                    split_rng,
                )
            )

            history_result = (
                evaluate_representation(
                    pair["history_name"],
                    2,
                    pair["history_length"],
                    train_indices,
                    test_indices,
                    all_moments,
                    all_targets,
                    split_rng,
                )
            )

            moment_result["budget"] = (
                pair["budget"]
            )

            history_result["budget"] = (
                pair["budget"]
            )

            moment_result["split"] = (
                split
            )

            history_result["split"] = (
                split
            )

            all_split_results.append(
                moment_result
            )

            all_split_results.append(
                history_result
            )

            print(
                f"  budget={pair['budget']:2d} "
                f"{pair['moment_name']:18s} "
                f"RMS={moment_result['test_rms']:.8e} "
                f"| "
                f"{pair['history_name']:20s} "
                f"RMS={history_result['test_rms']:.8e}"
            )

        print()

    # -----------------------------------------------------------------------
    # Aggregate split results
    # -----------------------------------------------------------------------

    representations = [
        pair["moment_name"]
        for pair in MATCHED_PAIRS
    ] + [
        pair["history_name"]
        for pair in MATCHED_PAIRS
    ]

    aggregate = []

    for name in representations:
        rows = [
            r
            for r in all_split_results
            if r["representation"] == name
        ]

        rms_values = np.asarray(
            [
                r["test_rms"]
                for r in rows
            ]
        )

        r2_values = np.asarray(
            [
                r["test_r2"]
                for r in rows
            ]
        )

        aggregate.append(
            {
                "representation": name,
                "budget": rows[0]["budget"],
                "features": rows[0]["features"],
                "parameters": rows[0]["parameters"],
                "mean_rms": float(
                    np.mean(
                        rms_values
                    )
                ),
                "std_rms": float(
                    np.std(
                        rms_values,
                        ddof=1,
                    )
                ),
                "mean_r2": float(
                    np.mean(
                        r2_values
                    )
                ),
            }
        )

    # -----------------------------------------------------------------------
    # Matched-pair statistics
    # -----------------------------------------------------------------------

    pair_results = []

    for pair in MATCHED_PAIRS:
        moment_rows = [
            r
            for r in all_split_results
            if (
                r["representation"]
                == pair["moment_name"]
            )
        ]

        history_rows = [
            r
            for r in all_split_results
            if (
                r["representation"]
                == pair["history_name"]
            )
        ]

        moment_rms = np.asarray(
            [
                r["test_rms"]
                for r in moment_rows
            ]
        )

        history_rms = np.asarray(
            [
                r["test_rms"]
                for r in history_rows
            ]
        )

        # Positive value means history has lower error.
        difference = (
            moment_rms
            - history_rms
        )

        relative_difference = (
            difference
            / moment_rms
        )

        pair_results.append(
            {
                "budget": pair["budget"],
                "moment_representation": pair[
                    "moment_name"
                ],
                "history_representation": pair[
                    "history_name"
                ],
                "moment_mean_rms": float(
                    np.mean(
                        moment_rms
                    )
                ),
                "history_mean_rms": float(
                    np.mean(
                        history_rms
                    )
                ),
                "mean_rms_difference": float(
                    np.mean(
                        difference
                    )
                ),
                "mean_fractional_advantage": float(
                    np.mean(
                        relative_difference
                    )
                ),
                "history_better_fraction": float(
                    np.mean(
                        difference > 0
                    )
                ),
            }
        )

    # -----------------------------------------------------------------------
    # Print aggregate results
    # -----------------------------------------------------------------------

    print()
    print(
        "============================================================"
    )
    print(
        "006n — Aggregate matched-budget results"
    )
    print(
        "============================================================"
    )
    print()

    for result in sorted(
        aggregate,
        key=lambda r: r["budget"],
    ):
        print(
            f"{result['representation']:20s} "
            f"budget={result['budget']:2d} "
            f"features={result['features']:2d} "
            f"parameters={result['parameters']:2d} "
            f"mean_RMS={result['mean_rms']:.8e} "
            f"std={result['std_rms']:.8e} "
            f"mean_R2={result['mean_r2']:.6f}"
        )

    print()
    print(
        "Matched-pair comparisons:"
    )
    print()

    for result in pair_results:
        print(
            f"budget={result['budget']:2d} "
            f"{result['moment_representation']:18s} "
            f"vs "
            f"{result['history_representation']:20s} "
            f"mean_RMS_difference="
            f"{result['mean_rms_difference']:.8e} "
            f"history_advantage="
            f"{result['mean_fractional_advantage']:.6f} "
            f"history_better_fraction="
            f"{result['history_better_fraction']:.3f}"
        )

    # -----------------------------------------------------------------------
    # Save detailed CSV
    # -----------------------------------------------------------------------

    csv_path = (
        RESULTS_DIR
        / "006n_equal_budget_memory_vs_moment.csv"
    )

    columns = [
        "split",
        "budget",
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

    with csv_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            ",".join(columns)
            + "\n"
        )

        for row in all_split_results:
            f.write(
                ",".join(
                    str(row[c])
                    for c in columns
                )
                + "\n"
            )

    # -----------------------------------------------------------------------
    # Save summary
    # -----------------------------------------------------------------------

    summary_path = (
        RESULTS_DIR
        / "006n_equal_budget_memory_vs_moment_summary.txt"
    )

    lines = [
        "Experiment 006n — Equal-Budget Memory vs Moment Closure",
        "",
        "Microscopic dynamics:",
        "    dx/dt = x + x^3 - x^5",
        "",
        "Question:",
        "    At comparable model capacity, does temporal history",
        "    provide more predictive information than additional",
        "    instantaneous moments?",
        "",
        f"Independent train/test splits: {N_SPLITS}",
        f"Train trajectories per split: {N_TRAIN}",
        f"Test trajectories per split: {N_TEST}",
        "",
        "Aggregate results:",
        "",
    ]

    for result in sorted(
        aggregate,
        key=lambda r: r["budget"],
    ):
        lines.append(
            f"{result['representation']:20s} "
            f"budget={result['budget']:2d} "
            f"features={result['features']:2d} "
            f"parameters={result['parameters']:2d} "
            f"mean_RMS={result['mean_rms']:.8e} "
            f"std_RMS={result['std_rms']:.8e} "
            f"mean_R2={result['mean_r2']:.6f}"
        )

    lines.extend(
        [
            "",
            "Matched-pair comparisons:",
            "",
        ]
    )

    for result in pair_results:
        lines.append(
            f"budget={result['budget']:2d} "
            f"{result['moment_representation']} "
            f"vs "
            f"{result['history_representation']} "
            f"mean_RMS_difference="
            f"{result['mean_rms_difference']:.8e} "
            f"history_fractional_advantage="
            f"{result['mean_fractional_advantage']:.6f} "
            f"history_better_fraction="
            f"{result['history_better_fraction']:.3f}"
        )

    lines.extend(
        [
            "",
            "Interpretation rule:",
            "    Positive RMS difference means the history representation",
            "    has lower test error than its matched moment representation.",
            "",
            "Important limitation:",
            "    Equal parameter count controls one important source of",
            "    model-capacity bias, but it does not prove that the two",
            "    representations contain equal information or equal",
            "    effective complexity.",
            "",
            "The m1...m6 representation is theoretically important because",
            "m1...m6 contain every moment appearing in the exact instantaneous",
            "derivatives of m1 and m2.",
        ]
    )

    summary_path.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )

    # -----------------------------------------------------------------------
    # Plot
    # -----------------------------------------------------------------------

    budgets = [
        pair["budget"]
        for pair in MATCHED_PAIRS
    ]

    moment_means = []
    history_means = []

    for budget in budgets:
        moment_result = next(
            r
            for r in aggregate
            if (
                r["budget"] == budget
                and r["representation"]
                == next(
                    p["moment_name"]
                    for p in MATCHED_PAIRS
                    if p["budget"] == budget
                )
            )
        )

        history_result = next(
            r
            for r in aggregate
            if (
                r["budget"] == budget
                and r["representation"]
                == next(
                    p["history_name"]
                    for p in MATCHED_PAIRS
                    if p["budget"] == budget
                )
            )
        )

        moment_means.append(
            moment_result["mean_rms"]
        )

        history_means.append(
            history_result["mean_rms"]
        )

    x = np.arange(
        len(budgets)
    )

    width = 0.36

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        x - width / 2,
        moment_means,
        width,
        label="Moment closure",
    )

    plt.bar(
        x + width / 2,
        history_means,
        width,
        label="History closure",
    )

    plt.xticks(
        x,
        [
            str(b)
            for b in budgets
        ],
    )

    plt.xlabel(
        "Matched parameter budget"
    )

    plt.ylabel(
        "Mean out-of-sample RMS error"
    )

    plt.title(
        "006n — Equal-budget memory vs moment closure"
    )

    plt.legend()

    plt.tight_layout()

    plot_path = (
        RESULTS_DIR
        / "006n_equal_budget_memory_vs_moment.png"
    )

    plt.savefig(
        plot_path,
        dpi=180,
    )

    plt.close()

    print()
    print(
        f"Saved: {csv_path}"
    )
    print(
        f"Saved: {summary_path}"
    )
    print(
        f"Saved: {plot_path}"
    )


if __name__ == "__main__":
    main()
