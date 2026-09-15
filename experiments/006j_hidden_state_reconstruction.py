"""
Experiment 006j: Hidden-State Reconstruction from Coarse History

Question
--------
If two microscopic states have identical instantaneous coarse variables
but different hidden microscopic structure, can their recent coarse
history reveal which hidden state they came from?

Microscopic dynamics
--------------------
    dx/dt = x + x^3 - x^5

Retained variables
------------------
    m1 = E[x]
    m2 = E[x^2]

Experiment
-----------
Construct two ensembles A and B with identical initial m1 and m2 but
different higher-order structure.

Generate microscopic trajectories and compare three predictors:

    1. Instantaneous state:
           H(t)

    2. Ordered coarse history:
           H(t), H(t-dt), ..., H(t-k*dt)

    3. Shuffled history:
           same historical values, but temporal ordering destroyed.

A useful memory effect should appear as:

    ordered history > shuffled history > instantaneous state

for hidden-state discrimination.

The classifier is deliberately simple: ridge logistic regression
implemented directly with NumPy.

This is a toy information-reconstruction experiment, not a claim
about fundamental physical memory.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

RNG = np.random.default_rng(24680)

N_PARTICLES = 16000

DT = 0.002
T_PRE = 4.0
T_POST = 4.0

# History lengths measured in samples.
HISTORY_LENGTHS = [0, 1, 2, 4, 8, 16]

RIDGE = 1e-3

TRAIN_PAIRS = 30
TEST_PAIRS = 15


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

    return x + (
        dt / 6.0
        * (k1 + 2*k2 + 2*k3 + k4)
    )


# ---------------------------------------------------------------------
# Coarse variables
# ---------------------------------------------------------------------

def retained_state(x):
    """
    Return H = (m1, m2).
    """

    return np.array([
        np.mean(x),
        np.mean(x**2),
    ])


# ---------------------------------------------------------------------
# Ensemble construction
# ---------------------------------------------------------------------

def make_ensemble_A():
    """
    Approximately Gaussian ensemble.
    """

    return RNG.normal(
        loc=1.0,
        scale=0.20,
        size=N_PARTICLES,
    )


def make_ensemble_B():
    """
    Non-Gaussian mixture.

    It is subsequently affine-normalised so that m1 and m2 match
    ensemble A as closely as possible.
    """

    components = RNG.choice(
        [0, 1, 2],
        size=N_PARTICLES,
        p=[0.25, 0.50, 0.25],
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

    return x


def match_first_two_moments(reference, candidate):
    """
    Affinely transform candidate so that its mean and variance match
    the reference exactly.
    """

    ref_mean = np.mean(reference)
    ref_std = np.std(reference)

    cand_mean = np.mean(candidate)
    cand_std = np.std(candidate)

    candidate = (
        (candidate - cand_mean)
        / cand_std
    )

    candidate = (
        candidate * ref_std
        + ref_mean
    )

    return candidate


# ---------------------------------------------------------------------
# History generation
# ---------------------------------------------------------------------

def generate_pair():
    """
    Generate a matched A/B microscopic pair.
    """

    A = make_ensemble_A()
    B = make_ensemble_B()

    B = match_first_two_moments(
        A,
        B,
    )

    return A, B


def simulate_history(x):
    """
    Simulate microscopic dynamics and return coarse history.

    The simulation contains a pre-history followed by a post-history
    interval. The pre-history provides the temporal context needed
    to reconstruct hidden state.

    Returns
    -------
    history : ndarray
        Shape (n_steps, 2)
    """

    n_steps = int(
        round(
            (T_PRE + T_POST) / DT
        )
    ) + 1

    history = np.zeros(
        (n_steps, 2)
    )

    for i in range(n_steps):

        history[i] = retained_state(x)

        if i < n_steps - 1:
            x = rk4_step(x, DT)

    return history


# ---------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------

def build_history_features(
    history,
    history_length,
    shuffle=False,
):
    """
    Construct features from a coarse trajectory.

    For each time t, use:

        H(t),
        H(t-1),
        ...
        H(t-k)

    If shuffle=True, historical ordering is destroyed while retaining
    the same collection of historical observations.
    """

    n = len(history)

    features = []
    labels = []

    start = history_length

    for i in range(start, n):

        window = history[
            i - history_length:
            i + 1
        ].copy()

        if shuffle and history_length > 1:

            # Preserve current H(t).
            current = window[-1].copy()

            past = window[:-1]

            RNG.shuffle(past)

            window = np.vstack([
                past,
                current,
            ])

        features.append(
            window.reshape(-1)
        )

    return np.asarray(features)


# ---------------------------------------------------------------------
# Build labelled pair datasets
# ---------------------------------------------------------------------

def generate_dataset(
    n_pairs,
    history_length,
    shuffled=False,
):
    """
    Generate A/B classification dataset.
    """

    X = []
    y = []

    for _ in range(n_pairs):

        A, B = generate_pair()

        history_A = simulate_history(A)
        history_B = simulate_history(B)

        features_A = build_history_features(
            history_A,
            history_length,
            shuffle=shuffled,
        )

        features_B = build_history_features(
            history_B,
            history_length,
            shuffle=shuffled,
        )

        X.append(features_A)
        y.append(
            np.zeros(
                len(features_A)
            )
        )

        X.append(features_B)
        y.append(
            np.ones(
                len(features_B)
            )
        )

    return (
        np.vstack(X),
        np.concatenate(y),
    )


# ---------------------------------------------------------------------
# Feature scaling
# ---------------------------------------------------------------------

def fit_scaling(X):
    mean = np.mean(
        X,
        axis=0,
    )

    scale = np.std(
        X,
        axis=0,
    )

    scale[
        scale < 1e-12
    ] = 1.0

    return mean, scale


def apply_scaling(
    X,
    mean,
    scale,
):
    return (
        (X - mean)
        / scale
    )


# ---------------------------------------------------------------------
# Logistic regression
# ---------------------------------------------------------------------

def sigmoid(z):
    """
    Numerically stable logistic function.
    """

    z = np.clip(
        z,
        -50.0,
        50.0,
    )

    return 1.0 / (
        1.0 + np.exp(-z)
    )


def fit_logistic(
    X,
    y,
    ridge=RIDGE,
    iterations=3000,
    learning_rate=0.05,
):
    """
    Simple ridge-regularised logistic regression.

    A bias column is added automatically.
    """

    X_aug = np.column_stack([
        np.ones(len(X)),
        X,
    ])

    n_features = X_aug.shape[1]

    weights = np.zeros(
        n_features
    )

    for _ in range(iterations):

        scores = X_aug @ weights

        probabilities = sigmoid(
            scores
        )

        gradient = (
            X_aug.T
            @ (probabilities - y)
            / len(y)
        )

        # Ridge penalty, excluding bias.
        gradient[1:] += (
            ridge
            * weights[1:]
        )

        weights -= (
            learning_rate
            * gradient
        )

    return weights


def predict_probability(
    X,
    weights,
):
    X_aug = np.column_stack([
        np.ones(len(X)),
        X,
    ])

    return sigmoid(
        X_aug @ weights
    )


def classification_accuracy(
    y,
    probabilities,
):
    predictions = (
        probabilities >= 0.5
    ).astype(float)

    return float(
        np.mean(
            predictions == y
        )
    )


# ---------------------------------------------------------------------
# Run one condition
# ---------------------------------------------------------------------

def run_condition(
    history_length,
    shuffled=False,
):
    """
    Train and test a classifier.
    """

    X_train, y_train = (
        generate_dataset(
            TRAIN_PAIRS,
            history_length,
            shuffled=shuffled,
        )
    )

    X_test, y_test = (
        generate_dataset(
            TEST_PAIRS,
            history_length,
            shuffled=shuffled,
        )
    )

    train_mean, train_scale = (
        fit_scaling(X_train)
    )

    X_train = apply_scaling(
        X_train,
        train_mean,
        train_scale,
    )

    X_test = apply_scaling(
        X_test,
        train_mean,
        train_scale,
    )

    weights = fit_logistic(
        X_train,
        y_train,
    )

    probabilities = predict_probability(
        X_test,
        weights,
    )

    accuracy = classification_accuracy(
        y_test,
        probabilities,
    )

    return accuracy


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def main():

    print(
        "Experiment 006j: "
        "Hidden-State Reconstruction from Coarse History"
    )

    print()

    print(
        "Microscopic dynamics:"
    )

    print(
        "    dx/dt = x + x^3 - x^5"
    )

    print()

    print(
        "Retained variables:"
    )

    print(
        "    m1 = E[x]"
    )

    print(
        "    m2 = E[x^2]"
    )

    print()

    print(
        "Question:"
    )

    print(
        "    Can coarse history reveal hidden microscopic state?"
    )

    print()

    # -------------------------------------------------------------
    # Verify initial matching.
    # -------------------------------------------------------------

    A, B = generate_pair()

    H_A = retained_state(A)
    H_B = retained_state(B)

    print(
        "Initial matched-state diagnostic:"
    )

    print(
        f"    |m1_A - m1_B| = "
        f"{abs(H_A[0] - H_B[0]):.12e}"
    )

    print(
        f"    |m2_A - m2_B| = "
        f"{abs(H_A[1] - H_B[1]):.12e}"
    )

    print()

    # -------------------------------------------------------------
    # Results.
    # -------------------------------------------------------------

    results = []

    for history_length in HISTORY_LENGTHS:

        print(
            f"Testing history length "
            f"{history_length}..."
        )

        ordered_accuracy = (
            run_condition(
                history_length,
                shuffled=False,
            )
        )

        shuffled_accuracy = (
            run_condition(
                history_length,
                shuffled=True,
            )
        )

        results.append(
            (
                history_length,
                ordered_accuracy,
                shuffled_accuracy,
            )
        )

        print(
            f"    ordered accuracy  = "
            f"{ordered_accuracy:.6f}"
        )

        print(
            f"    shuffled accuracy = "
            f"{shuffled_accuracy:.6f}"
        )

    # -------------------------------------------------------------
    # Summary.
    # -------------------------------------------------------------

    best = max(
        results,
        key=lambda r: r[1],
    )

    best_history = best[0]
    best_ordered = best[1]
    best_shuffled = best[2]

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    print(
        f"Best ordered-history length = "
        f"{best_history}"
    )

    print(
        f"Best ordered accuracy = "
        f"{best_ordered:.6f}"
    )

    print(
        f"Corresponding shuffled accuracy = "
        f"{best_shuffled:.6f}"
    )

    print(
        f"Ordered-history advantage = "
        f"{best_ordered - best_shuffled:.6f}"
    )

    print()

    print(
        "Interpretation:"
    )

    print(
        "If ordered-history accuracy rises above "
        "instantaneous-state accuracy, coarse history "
        "contains information about hidden microscopic state."
    )

    print(
        "If ordered history also outperforms shuffled "
        "history, temporal ordering itself carries "
        "information rather than merely the collection "
        "of past coarse values."
    )

    print(
        "Chance-level classification is approximately 0.5."
    )

    # -------------------------------------------------------------
    # Save CSV.
    # -------------------------------------------------------------

    csv_path = (
        OUTPUT_DIR
        / "006j_hidden_state_reconstruction.csv"
    )

    with open(
        csv_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "history_length,"
            "ordered_accuracy,"
            "shuffled_accuracy\n"
        )

        for (
            history_length,
            ordered_accuracy,
            shuffled_accuracy,
        ) in results:

            f.write(
                f"{history_length},"
                f"{ordered_accuracy:.12f},"
                f"{shuffled_accuracy:.12f}\n"
            )

    # -------------------------------------------------------------
    # Save summary.
    # -------------------------------------------------------------

    summary_path = (
        OUTPUT_DIR
        / "006j_hidden_state_reconstruction_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "Experiment 006j: "
            "Hidden-State Reconstruction from Coarse History\n\n"
        )

        f.write(
            "Question:\n"
        )

        f.write(
            "Can the history of retained variables reveal "
            "hidden microscopic state?\n\n"
        )

        f.write(
            "Results:\n"
        )

        for (
            history_length,
            ordered_accuracy,
            shuffled_accuracy,
        ) in results:

            f.write(
                f"history={history_length:2d} "
                f"ordered={ordered_accuracy:.8f} "
                f"shuffled={shuffled_accuracy:.8f}\n"
            )

        f.write(
            "\nBest ordered history:\n"
        )

        f.write(
            f"    length = {best_history}\n"
        )

        f.write(
            f"    accuracy = {best_ordered:.8f}\n"
        )

        f.write(
            f"    shuffled = {best_shuffled:.8f}\n"
        )

        f.write(
            "\nOrdered-history advantage:\n"
        )

        f.write(
            f"    {best_ordered - best_shuffled:.8f}\n"
        )

        f.write(
            "\nInterpretation:\n"
        )

        f.write(
            "The experiment tests whether temporal information "
            "in a coarse representation can partially reconstruct "
            "hidden microscopic state.\n"
        )

    # -------------------------------------------------------------
    # Plot.
    # -------------------------------------------------------------

    history_values = [
        r[0]
        for r in results
    ]

    ordered_values = [
        r[1]
        for r in results
    ]

    shuffled_values = [
        r[2]
        for r in results
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history_values,
        ordered_values,
        marker="o",
        label="Ordered history",
    )

    plt.plot(
        history_values,
        shuffled_values,
        marker="s",
        label="Shuffled history",
    )

    plt.axhline(
        0.5,
        linestyle="--",
        label="Chance",
    )

    plt.xlabel(
        "History length"
    )

    plt.ylabel(
        "Hidden-state classification accuracy"
    )

    plt.title(
        "006j: Hidden-State Reconstruction from Coarse History"
    )

    plt.legend()
    plt.grid(True)

    plot_path = (
        OUTPUT_DIR
        / "006j_hidden_state_reconstruction.png"
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
    print(f"    {summary_path}")
    print(f"    {plot_path}")


if __name__ == "__main__":
    main()
