"""
Experiment 006k: Conditional Closure Uncertainty

Question
--------
How much uncertainty remains in the future coarse dynamics after
specifying the coarse state, and how much of that uncertainty can
be removed by adding temporal history?

Microscopic dynamics
--------------------
    dx/dt = x + x^3 - x^5

Retained variables
------------------
    m1 = E[x]
    m2 = E[x^2]

For many independently generated microscopic trajectories we record:

    H(t)    = (m1, m2)
    dH/dt

We estimate conditional uncertainty using nearest neighbours.

Instantaneous conditioning:

    U0 ~ Var[dH/dt | H(t)]

History conditioning:

    Uk ~ Var[dH/dt | H(t), H(t-dt), ..., H(t-k*dt)]

The experiment asks whether adding history makes the coarse state
more dynamically predictive.

This is a statistical diagnostic, not a claim about fundamental
physical memory.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(13579)

N_PARTICLES = 10000

DT = 0.002
T = 6.0

N_TRAJECTORIES = 80

# Number of samples in the history vector.
HISTORY_LENGTHS = [0, 1, 2, 4, 8, 16]

# Number of nearest neighbours used to estimate conditional variance.
K_NEIGHBOURS = 30

# Subsample trajectory points to keep the nearest-neighbour calculation
# manageable.
SAMPLE_STRIDE = 10


# ---------------------------------------------------------------------
# Microscopic dynamics
# ---------------------------------------------------------------------

def microscopic_rhs(x):
    """
    dx/dt = x + x^3 - x^5
    """

    return (
        x
        + x**3
        - x**5
    )


def rk4_step(x, dt):
    """
    Fourth-order Runge-Kutta step.
    """

    k1 = microscopic_rhs(x)

    k2 = microscopic_rhs(
        x + 0.5 * dt * k1
    )

    k3 = microscopic_rhs(
        x + 0.5 * dt * k2
    )

    k4 = microscopic_rhs(
        x + dt * k3
    )

    return x + (
        dt
        / 6.0
        * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        )
    )


# ---------------------------------------------------------------------
# Initial microscopic states
# ---------------------------------------------------------------------

def generate_initial_state(kind):
    """
    Generate different microscopic ensembles.

    The ensembles deliberately differ in higher-order structure.
    """

    if kind == 0:

        x = RNG.normal(
            1.0,
            0.20,
            N_PARTICLES,
        )

    elif kind == 1:

        components = RNG.choice(
            [0, 1, 2],
            size=N_PARTICLES,
            p=[
                0.25,
                0.50,
                0.25,
            ],
        )

        means = np.array([
            0.70,
            1.00,
            1.30,
        ])

        x = (
            means[components]
            + RNG.normal(
                0.0,
                0.08,
                N_PARTICLES,
            )
        )

    elif kind == 2:

        x = RNG.normal(
            0.90,
            0.28,
            N_PARTICLES,
        )

    elif kind == 3:

        x = RNG.uniform(
            0.45,
            1.55,
            N_PARTICLES,
        )

    else:

        z = RNG.normal(
            0.0,
            1.0,
            N_PARTICLES,
        )

        x = (
            1.0
            + 0.18 * z
            + 0.04 * (
                z**2 - 1.0
            )
        )

    return x


# ---------------------------------------------------------------------
# Coarse variables
# ---------------------------------------------------------------------

def retained_state(x):
    """
    H = (m1, m2)
    """

    return np.array([
        np.mean(x),
        np.mean(x**2),
    ])


def retained_derivative(x):
    """
    Exact microscopic derivative of H.

    dm1/dt = E[x + x^3 - x^5]

    dm2/dt = 2 E[x^2 + x^4 - x^6]
    """

    dm1 = np.mean(
        x
        + x**3
        - x**5
    )

    dm2 = 2.0 * np.mean(
        x**2
        + x**4
        - x**6
    )

    return np.array([
        dm1,
        dm2,
    ])


# ---------------------------------------------------------------------
# Simulate one trajectory
# ---------------------------------------------------------------------

def simulate_trajectory(kind):
    """
    Generate one coarse trajectory and its exact coarse derivative.
    """

    n_steps = (
        int(
            round(T / DT)
        )
        + 1
    )

    history = np.zeros(
        (n_steps, 2)
    )

    derivative = np.zeros(
        (n_steps, 2)
    )

    x = generate_initial_state(
        kind
    )

    for i in range(n_steps):

        history[i] = retained_state(
            x
        )

        derivative[i] = retained_derivative(
            x
        )

        if i < n_steps - 1:

            x = rk4_step(
                x,
                DT,
            )

    return history, derivative


# ---------------------------------------------------------------------
# Build global dataset
# ---------------------------------------------------------------------

def build_dataset():
    """
    Generate many independent microscopic trajectories.

    Returns
    -------
    histories : ndarray
        Shape (N, 2)

    derivatives : ndarray
        Shape (N, 2)

    trajectory_ids : ndarray
        Identifies the originating microscopic trajectory.
    """

    histories = []
    derivatives = []
    trajectory_ids = []

    print(
        "Generating microscopic trajectories..."
    )

    for trajectory in range(
        N_TRAJECTORIES
    ):

        kind = trajectory % 5

        history, derivative = (
            simulate_trajectory(
                kind
            )
        )

        # Subsample.
        indices = np.arange(
            0,
            len(history),
            SAMPLE_STRIDE,
        )

        histories.append(
            history[indices]
        )

        derivatives.append(
            derivative[indices]
        )

        trajectory_ids.append(
            np.full(
                len(indices),
                trajectory,
                dtype=int,
            )
        )

    histories = np.vstack(
        histories
    )

    derivatives = np.vstack(
        derivatives
    )

    trajectory_ids = np.concatenate(
        trajectory_ids
    )

    return (
        histories,
        derivatives,
        trajectory_ids,
    )


# ---------------------------------------------------------------------
# Construct history vectors
# ---------------------------------------------------------------------

def construct_features(
    histories,
    trajectory_ids,
    history_length,
):
    """
    Construct current-state or history features.

    Important:
        History never crosses a trajectory boundary.
    """

    features = []
    derivatives_indices = []

    for trajectory in np.unique(
        trajectory_ids
    ):

        indices = np.where(
            trajectory_ids
            == trajectory
        )[0]

        # Because each trajectory was appended contiguously,
        # indices are ordered correctly.
        indices = indices[
            history_length:
        ]

        for position, global_index in enumerate(
            indices,
            start=history_length,
        ):

            local_start = (
                position
                - history_length
            )

            local_indices = (
                np.where(
                    trajectory_ids
                    == trajectory
                )[0]
            )

            selected = local_indices[
                local_start:
                position + 1
            ]

            values = histories[
                selected
            ]

            features.append(
                values.reshape(-1)
            )

            derivatives_indices.append(
                global_index
            )

    return (
        np.asarray(features),
        np.asarray(
            derivatives_indices,
            dtype=int,
        ),
    )


# ---------------------------------------------------------------------
# Standardisation
# ---------------------------------------------------------------------

def standardise(values):
    mean = np.mean(
        values,
        axis=0,
    )

    scale = np.std(
        values,
        axis=0,
    )

    scale[
        scale < 1e-12
    ] = 1.0

    return (
        (values - mean) / scale,
        mean,
        scale,
    )


# ---------------------------------------------------------------------
# Nearest-neighbour conditional variance
# ---------------------------------------------------------------------

def conditional_uncertainty(
    features,
    derivatives,
    k=K_NEIGHBOURS,
):
    """
    Estimate conditional uncertainty using nearest neighbours.

    For each point:

        1. Find k nearby feature vectors.
        2. Measure variance of their derivative vectors.
        3. Average the variance.

    The returned quantity is the RMS scale of the conditional
    derivative variation.
    """

    n = len(features)

    # Standardise feature dimensions so m1 and m2, and their history,
    # contribute comparably.
    X, _, _ = standardise(
        features
    )

    # Standardise derivative components separately.
    Y, _, _ = standardise(
        derivatives
    )

    uncertainties = []

    for i in range(n):

        distances = np.sum(
            (
                X
                - X[i]
            )**2,
            axis=1,
        )

        # Exclude self.
        distances[i] = np.inf

        neighbour_indices = np.argpartition(
            distances,
            k,
        )[:k]

        neighbour_values = Y[
            neighbour_indices
        ]

        local_mean = np.mean(
            neighbour_values,
            axis=0,
        )

        local_variance = np.mean(
            (
                neighbour_values
                - local_mean
            )**2
        )

        uncertainties.append(
            local_variance
        )

    return float(
        np.sqrt(
            np.mean(
                uncertainties
            )
        )
    )


# ---------------------------------------------------------------------
# Bootstrap uncertainty estimate
# ---------------------------------------------------------------------

def bootstrap_uncertainty(
    features,
    derivatives,
    repetitions=8,
):
    """
    Estimate variability of the conditional-uncertainty statistic
    through bootstrap resampling.
    """

    n = len(features)

    values = []

    for _ in range(
        repetitions
    ):

        indices = RNG.integers(
            0,
            n,
            size=n,
        )

        values.append(
            conditional_uncertainty(
                features[indices],
                derivatives[indices],
            )
        )

    return (
        float(np.mean(values)),
        float(np.std(values)),
    )


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():

    print(
        "Experiment 006k: "
        "Conditional Closure Uncertainty"
    )

    print()

    print(
        "Question:"
    )

    print(
        "    How much uncertainty remains in the coarse "
        "dynamics after specifying the coarse state?"
    )

    print()

    (
        histories,
        derivatives,
        trajectory_ids,
    ) = build_dataset()

    print()

    print(
        f"Dataset size: {len(histories)} points"
    )

    print()

    results = []

    for history_length in HISTORY_LENGTHS:

        print(
            f"Evaluating history length "
            f"{history_length}..."
        )

        (
            features,
            derivative_indices,
        ) = construct_features(
            histories,
            trajectory_ids,
            history_length,
        )

        target_derivatives = (
            derivatives[
                derivative_indices
            ]
        )

        # Reduce the dataset for expensive nearest-neighbour
        # calculations while preserving the distribution.
        if len(features) > 12000:

            selected = RNG.choice(
                len(features),
                size=12000,
                replace=False,
            )

            features_eval = features[
                selected
            ]

            derivatives_eval = (
                target_derivatives[
                    selected
                ]
            )

        else:

            features_eval = features
            derivatives_eval = (
                target_derivatives
            )

        uncertainty, bootstrap_std = (
            bootstrap_uncertainty(
                features_eval,
                derivatives_eval,
                repetitions=8,
            )
        )

        results.append(
            (
                history_length,
                uncertainty,
                bootstrap_std,
            )
        )

        print(
            f"    conditional uncertainty = "
            f"{uncertainty:.10e}"
        )

        print(
            f"    bootstrap std = "
            f"{bootstrap_std:.10e}"
        )

    # -------------------------------------------------------------
    # Compare against instantaneous case.
    # -------------------------------------------------------------

    baseline = results[0][1]

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    print(
        f"Instantaneous uncertainty = "
        f"{baseline:.10e}"
    )

    for (
        history_length,
        uncertainty,
        bootstrap_std,
    ) in results:

        reduction = (
            baseline
            - uncertainty
        )

        fractional = (
            reduction
            / baseline
        )

        print(
            f"history={history_length:2d} "
            f"uncertainty={uncertainty:.10e} "
            f"reduction={fractional:.6f}"
        )

    # -------------------------------------------------------------
    # Save CSV.
    # -------------------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "006k_conditional_uncertainty.csv"
    )

    with open(
        csv_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "history_length,"
            "conditional_uncertainty,"
            "bootstrap_std,"
            "fractional_reduction\n"
        )

        for (
            history_length,
            uncertainty,
            bootstrap_std,
        ) in results:

            fractional = (
                baseline
                - uncertainty
            ) / baseline

            f.write(
                f"{history_length},"
                f"{uncertainty:.12e},"
                f"{bootstrap_std:.12e},"
                f"{fractional:.12e}\n"
            )

    # -------------------------------------------------------------
    # Save report.
    # -------------------------------------------------------------

    report_path = (
        OUTPUT_DIR
        / "006k_conditional_uncertainty_summary.txt"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Experiment 006k: "
            "Conditional Closure Uncertainty\n\n"
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
            "Results:\n"
        )

        for (
            history_length,
            uncertainty,
            bootstrap_std,
        ) in results:

            fractional = (
                baseline
                - uncertainty
            ) / baseline

            f.write(
                f"history={history_length:2d} "
                f"uncertainty={uncertainty:.12e} "
                f"bootstrap_std={bootstrap_std:.12e} "
                f"fractional_reduction={fractional:.12e}\n"
            )

        f.write(
            "\nInterpretation:\n"
        )

        f.write(
            "Conditional uncertainty measures how much "
            "variation remains in the microscopic coarse "
            "derivative after conditioning on the retained "
            "state or its history.\n"
        )

        f.write(
            "A systematic reduction with history would "
            "indicate that temporal information makes the "
            "coarse representation more dynamically closed.\n"
        )

        f.write(
            "No reduction would indicate that the missing "
            "information is not recoverable from the tested "
            "coarse history.\n"
        )

    # -------------------------------------------------------------
    # Plot.
    # -------------------------------------------------------------

    x_values = [
        r[0]
        for r in results
    ]

    y_values = [
        r[1]
        for r in results
    ]

    y_errors = [
        r[2]
        for r in results
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.errorbar(
        x_values,
        y_values,
        yerr=y_errors,
        marker="o",
        capsize=4,
    )

    plt.xlabel(
        "History length"
    )

    plt.ylabel(
        "Conditional derivative uncertainty"
    )

    plt.title(
        "006k: Conditional Closure Uncertainty"
    )

    plt.grid(True)

    plot_path = (
        OUTPUT_DIR
        / "006k_conditional_uncertainty.png"
    )

    plt.tight_layout()
    plt.savefig(
        plot_path,
        dpi=160,
    )

    plt.close()

    print()
    print("Saved:")
    print(f"    {csv_path}")
    print(f"    {report_path}")
    print(f"    {plot_path}")


if __name__ == "__main__":
    main()
