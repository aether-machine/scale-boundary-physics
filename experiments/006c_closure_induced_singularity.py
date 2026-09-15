"""
Experiment 006c: Closure-Induced Singularity
=============================================

Question
--------
Can a completely regular underlying dynamical system produce a
finite-time singularity when an insufficient effective closure is used?

This experiment is deliberately different from 006b.

The microscopic/underlying system is

    dx/dt = y

    epsilon * dy/dt = g(x) - y

with

    g(x) = x + x^3 - x^5.

The quintic term stabilises the dynamics for large |x|.

The complete system therefore has a strong restoring force at large
amplitude and remains bounded.

For small/moderate x, however, the dynamics are approximately

    g(x) ~ x + x^3.

We construct an effective cubic closure from data in a restricted
training region and then extrapolate it.

The resulting effective model is

    dx/dt = a1*x + a3*x^3.

If a3 > 0, this first-order effective model can develop a finite-time
blow-up even though the complete underlying system remains bounded.

This is NOT intended to demonstrate that Navier-Stokes behaves this way.

It tests a narrower proposition:

    Can loss of higher-order stabilising structure during closure
    produce a singular effective equation?

The distinction between:
    - underlying dynamics,
    - observed/reduced dynamics,
    - closure,
    - extrapolation,
is central to the experiment.

Outputs
-------
results/006c_closure_singularity.csv
results/006c_closure_singularity_summary.txt

results/006c_underlying_vs_closure.png
results/006c_closure_fit.png
results/006c_blowup_comparison.png
"""


from __future__ import annotations


import csv
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")


DT = 0.001
T_FINAL = 20.0


# Fast relaxation parameter.
#
# Smaller epsilon means y tracks g(x) more closely, making the effective
# first-order interpretation more appropriate.
EPSILON_VALUES = [
    0.05,
    0.10,
    0.20,
]


# Training region.
#
# We intentionally do NOT train on the large-|x| regime where the
# stabilising -x^5 term dominates.
TRAIN_LIMIT = 1.0


# Test initial condition.
#
# This lies just outside the main training region.
TEST_X0 = 1.10
TEST_Y0 = 0.0


# Polynomial closure degrees.
CLOSURE_DEGREES = [
    1,
    3,
    5,
]


# Safety limit for numerical runaway.
SAFETY_LIMIT = 1.0e6


RIDGE = 1.0e-10


# ---------------------------------------------------------------------------
# Underlying nonlinear system
# ---------------------------------------------------------------------------

def g(
    x: float | np.ndarray,
) -> float | np.ndarray:
    """
    Complete nonlinear restoring dynamics.

        g(x) = x + x^3 - x^5

    The negative quintic term dominates at large |x| and prevents
    indefinite growth.
    """

    return (
        x
        + x**3
        - x**5
    )


def underlying_rhs(
    x: float,
    y: float,
    epsilon: float,
) -> tuple[float, float]:
    """
    Complete two-variable system.

        dx/dt = y

        epsilon * dy/dt = g(x) - y
    """

    dxdt = y

    dydt = (
        g(x) - y
    ) / epsilon

    return (
        dxdt,
        dydt,
    )


def rk4_underlying_step(
    x: float,
    y: float,
    dt: float,
    epsilon: float,
) -> tuple[float, float]:

    k1x, k1y = underlying_rhs(
        x,
        y,
        epsilon,
    )

    k2x, k2y = underlying_rhs(
        x + 0.5 * dt * k1x,
        y + 0.5 * dt * k1y,
        epsilon,
    )

    k3x, k3y = underlying_rhs(
        x + 0.5 * dt * k2x,
        y + 0.5 * dt * k2y,
        epsilon,
    )

    k4x, k4y = underlying_rhs(
        x + dt * k3x,
        y + dt * k3y,
        epsilon,
    )

    x_new = (
        x
        + (dt / 6.0)
        * (
            k1x
            + 2.0 * k2x
            + 2.0 * k3x
            + k4x
        )
    )

    y_new = (
        y
        + (dt / 6.0)
        * (
            k1y
            + 2.0 * k2y
            + 2.0 * k3y
            + k4y
        )
    )

    return (
        x_new,
        y_new,
    )


def integrate_underlying(
    x0: float,
    y0: float,
    epsilon: float,
    dt: float,
    t_final: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:

    n_steps = int(
        round(t_final / dt)
    )

    times = (
        np.arange(
            n_steps + 1,
            dtype=float,
        )
        * dt
    )

    x = np.empty(
        n_steps + 1
    )

    y = np.empty(
        n_steps + 1
    )

    x[0] = x0
    y[0] = y0

    for i in range(n_steps):

        x[i + 1], y[i + 1] = (
            rk4_underlying_step(
                x[i],
                y[i],
                dt,
                epsilon,
            )
        )

    return (
        times,
        x,
        y,
    )


# ---------------------------------------------------------------------------
# Generate closure-training data
# ---------------------------------------------------------------------------

def build_training_data(
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:

    x_samples = []
    dxdt_samples = []

    # Several initial conditions inside the training region.
    initial_conditions = np.linspace(
        -0.9,
        0.9,
        13,
    )

    training_time = 8.0

    for x0 in initial_conditions:

        # Start approximately on the slow manifold.
        y0 = float(
            g(x0)
        )

        times, x, y = (
            integrate_underlying(
                x0,
                y0,
                epsilon,
                DT,
                training_time,
            )
        )

        # Only retain points inside the training domain.
        mask = (
            np.abs(x)
            <= TRAIN_LIMIT
        )

        x_samples.append(
            x[mask]
        )

        # dx/dt = y exactly.
        dxdt_samples.append(
            y[mask]
        )

    return (
        np.concatenate(
            x_samples
        ),
        np.concatenate(
            dxdt_samples
        ),
    )


# ---------------------------------------------------------------------------
# Polynomial closure
# ---------------------------------------------------------------------------

def design_matrix(
    x: np.ndarray,
    degree: int,
) -> np.ndarray:

    return np.column_stack(
        [
            x**power
            for power in range(
                degree + 1
            )
        ]
    )


def fit_closure(
    x: np.ndarray,
    dxdt: np.ndarray,
    degree: int,
) -> np.ndarray:

    A = design_matrix(
        x,
        degree,
    )

    ATA = (
        A.T @ A
    )

    regularisation = (
        RIDGE
        * np.eye(
            ATA.shape[0]
        )
    )

    coefficients = np.linalg.solve(
        ATA
        + regularisation,
        A.T @ dxdt,
    )

    return coefficients


def evaluate_closure(
    x: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:

    result = np.zeros_like(
        x,
        dtype=float,
    )

    for power, coefficient in enumerate(
        coefficients
    ):

        result += (
            coefficient
            * x**power
        )

    return result


# ---------------------------------------------------------------------------
# Effective closure integration
# ---------------------------------------------------------------------------

def closure_rhs(
    x: float,
    coefficients: np.ndarray,
) -> float:

    return float(
        evaluate_closure(
            np.array([x]),
            coefficients,
        )[0]
    )


def rk4_closure_step(
    x: float,
    dt: float,
    coefficients: np.ndarray,
) -> float:

    k1 = closure_rhs(
        x,
        coefficients,
    )

    k2 = closure_rhs(
        x + 0.5 * dt * k1,
        coefficients,
    )

    k3 = closure_rhs(
        x + 0.5 * dt * k2,
        coefficients,
    )

    k4 = closure_rhs(
        x + dt * k3,
        coefficients,
    )

    return (
        x
        + (dt / 6.0)
        * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        )
    )


def integrate_closure(
    x0: float,
    coefficients: np.ndarray,
    dt: float,
    t_final: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float | None,
]:

    n_steps = int(
        round(t_final / dt)
    )

    times = (
        np.arange(
            n_steps + 1,
            dtype=float,
        )
        * dt
    )

    x = np.full(
        n_steps + 1,
        np.nan,
    )

    x[0] = x0

    failure_time = None

    for i in range(n_steps):

        if (
            not np.isfinite(
                x[i]
            )
            or abs(x[i])
            > SAFETY_LIMIT
        ):

            failure_time = times[i]
            break

        x_next = rk4_closure_step(
            x[i],
            dt,
            coefficients,
        )

        x[i + 1] = x_next

        if (
            not np.isfinite(
                x_next
            )
            or abs(x_next)
            > SAFETY_LIMIT
        ):

            failure_time = times[i + 1]
            break

    return (
        times,
        x,
        failure_time,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def rms(
    values: np.ndarray,
) -> float:

    return float(
        np.sqrt(
            np.mean(
                values**2
            )
        )
    )


def first_threshold_crossing(
    times: np.ndarray,
    values: np.ndarray,
    threshold: float,
) -> float | None:

    mask = (
        np.isfinite(values)
        & (
            np.abs(values)
            >= threshold
        )
    )

    indices = np.where(
        mask
    )[0]

    if len(indices) == 0:
        return None

    return float(
        times[indices[0]]
    )


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main() -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(
        "Experiment 006c: Closure-Induced Singularity"
    )
    print("=" * 72)
    print()

    print(
        "Underlying system:"
    )

    print(
        "    dx/dt = y"
    )

    print(
        "    epsilon * dy/dt = x + x^3 - x^5 - y"
    )

    print()

    print(
        "The complete system contains the stabilising -x^5 term."
    )

    print(
        "The effective closure will be fitted only inside"
        f" |x| <= {TRAIN_LIMIT}."
    )

    print()

    all_results = []

    model_store = {}

    # ---------------------------------------------------------------
    # Train closures for each epsilon
    # ---------------------------------------------------------------

    for epsilon in EPSILON_VALUES:

        print(
            f"epsilon = {epsilon:.3f}"
        )

        x_train, dxdt_train = (
            build_training_data(
                epsilon
            )
        )

        print(
            f"  training samples = "
            f"{len(x_train)}"
        )

        for degree in CLOSURE_DEGREES:

            coefficients = fit_closure(
                x_train,
                dxdt_train,
                degree,
            )

            fitted = evaluate_closure(
                x_train,
                coefficients,
            )

            training_error = rms(
                fitted
                - dxdt_train
            )

            model_store[
                (
                    epsilon,
                    degree,
                )
            ] = coefficients

            print(
                f"  degree={degree}: "
                f"training RMS="
                f"{training_error:.8e}"
            )

            all_results.append(
                {
                    "epsilon": epsilon,
                    "degree": degree,
                    "training_rms": training_error,
                }
            )

        print()

    # ---------------------------------------------------------------
    # Test against complete underlying dynamics
    # ---------------------------------------------------------------

    print(
        "Testing at:"
    )

    print(
        f"  x(0) = {TEST_X0}"
    )

    print(
        f"  y(0) = {TEST_Y0}"
    )

    print()

    for epsilon in EPSILON_VALUES:

        (
            times_true,
            x_true,
            y_true,
        ) = integrate_underlying(
            TEST_X0,
            TEST_Y0,
            epsilon,
            DT,
            T_FINAL,
        )

        print(
            f"epsilon = {epsilon:.3f}"
        )

        print(
            f"  underlying max|x| = "
            f"{np.max(np.abs(x_true)):.8f}"
        )

        print(
            f"  underlying max|y| = "
            f"{np.max(np.abs(y_true)):.8f}"
        )

        for degree in CLOSURE_DEGREES:

            coefficients = model_store[
                (
                    epsilon,
                    degree,
                )
            ]

            (
                times_model,
                x_model,
                failure,
            ) = integrate_closure(
                TEST_X0,
                coefficients,
                DT,
                T_FINAL,
            )

            valid = np.isfinite(
                x_model
            )

            common = (
                valid
                & np.isfinite(
                    x_true
                )
            )

            if np.any(common):

                prediction_error = rms(
                    x_model[common]
                    - x_true[common]
                )

                max_prediction_error = float(
                    np.max(
                        np.abs(
                            x_model[common]
                            - x_true[common]
                        )
                    )
                )

                max_model = float(
                    np.max(
                        np.abs(
                            x_model[common]
                        )
                    )
                )

            else:

                prediction_error = np.inf
                max_prediction_error = np.inf
                max_model = np.inf

            threshold_time = (
                first_threshold_crossing(
                    times_model,
                    x_model,
                    10.0,
                )
            )

            failure_text = (
                "none"
                if failure is None
                else f"{failure:.6f}"
            )

            threshold_text = (
                "none"
                if threshold_time is None
                else f"{threshold_time:.6f}"
            )

            print(
                f"  degree={degree}: "
                f"prediction RMS="
                f"{prediction_error:.8f}, "
                f"max error="
                f"{max_prediction_error:.8f}, "
                f"max|x|="
                f"{max_model:.8f}, "
                f"|x|>10="
                f"{threshold_text}, "
                f"failure="
                f"{failure_text}"
            )

            for row in all_results:

                if (
                    row["epsilon"]
                    == epsilon
                    and row["degree"]
                    == degree
                ):

                    row[
                        "prediction_rms"
                    ] = prediction_error

                    row[
                        "prediction_max"
                    ] = max_prediction_error

                    row[
                        "model_max_abs_x"
                    ] = max_model

                    row[
                        "threshold_10_time"
                    ] = threshold_time

                    row[
                        "failure_time"
                    ] = failure

        print()

    # ---------------------------------------------------------------
    # Write CSV
    # ---------------------------------------------------------------

    csv_path = (
        RESULTS_DIR
        / "006c_closure_singularity.csv"
    )

    fieldnames = [
        "epsilon",
        "degree",
        "training_rms",
        "prediction_rms",
        "prediction_max",
        "model_max_abs_x",
        "threshold_10_time",
        "failure_time",
    ]

    with csv_path.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in all_results:

            writer.writerow(
                {
                    field: row.get(
                        field,
                        "",
                    )
                    for field in fieldnames
                }
            )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    summary_path = (
        RESULTS_DIR
        / "006c_closure_singularity_summary.txt"
    )

    with summary_path.open(
        "w"
    ) as handle:

        handle.write(
            "Experiment 006c: Closure-Induced Singularity\n"
        )

        handle.write(
            "=" * 72 + "\n\n"
        )

        handle.write(
            "Question:\n"
        )

        handle.write(
            "Can a completely regular underlying dynamical system "
            "produce a finite-time singularity when an insufficient "
            "effective closure is used?\n\n"
        )

        handle.write(
            "Underlying system:\n"
        )

        handle.write(
            "    dx/dt = y\n"
        )

        handle.write(
            "    epsilon * dy/dt = x + x^3 - x^5 - y\n\n"
        )

        handle.write(
            "The -x^5 term stabilises the complete system at large |x|.\n\n"
        )

        handle.write(
            "Training region:\n"
        )

        handle.write(
            f"    |x| <= {TRAIN_LIMIT}\n\n"
        )

        handle.write(
            "Test initial condition:\n"
        )

        handle.write(
            f"    x(0) = {TEST_X0}\n"
        )

        handle.write(
            f"    y(0) = {TEST_Y0}\n\n"
        )

        handle.write(
            "Results:\n\n"
        )

        for row in all_results:

            handle.write(
                f"epsilon={row['epsilon']:.3f}, "
                f"degree={row['degree']}: "
                f"training_RMS="
                f"{row['training_rms']:.8e}, "
                f"prediction_RMS="
                f"{row.get('prediction_rms', np.nan):.8e}, "
                f"max|x|="
                f"{row.get('model_max_abs_x', np.nan):.8e}, "
                f"|x|>10="
                f"{row.get('threshold_10_time', None)}, "
                f"failure="
                f"{row.get('failure_time', None)}\n"
            )

        handle.write(
            "\nInterpretation:\n"
        )

        handle.write(
            "The complete underlying dynamics are regular because "
            "the negative quintic term dominates at large amplitude.\n\n"
        )

        handle.write(
            "The experiment asks whether an effective closure trained "
            "only in the moderate-amplitude regime can discard the "
            "higher-order stabilising structure.\n\n"
        )

        handle.write(
            "If a low-order closure develops finite-time runaway while "
            "the complete system remains bounded, the result demonstrates "
            "a representation-level singularity generated by closure "
            "rather than by the underlying dynamics.\n\n"
        )

        handle.write(
            "However, this would still NOT demonstrate that projection "
            "alone causes singularities. The crucial ingredient would be "
            "the approximation/truncation used in constructing the "
            "effective closure.\n"
        )

    # ---------------------------------------------------------------
    # Plot 1: complete underlying dynamics vs closure
    # ---------------------------------------------------------------

    epsilon_plot = EPSILON_VALUES[1]

    (
        times_true,
        x_true,
        y_true,
    ) = integrate_underlying(
        TEST_X0,
        TEST_Y0,
        epsilon_plot,
        DT,
        T_FINAL,
    )

    fig = plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        times_true,
        x_true,
        label="complete underlying x(t)",
        linewidth=2,
    )

    for degree in CLOSURE_DEGREES:

        coefficients = model_store[
            (
                epsilon_plot,
                degree,
            )
        ]

        (
            times_model,
            x_model,
            failure,
        ) = integrate_closure(
            TEST_X0,
            coefficients,
            DT,
            T_FINAL,
        )

        plt.plot(
            times_model,
            x_model,
            label=f"degree {degree} closure",
        )

    plt.xlabel("time")
    plt.ylabel("x(t)")
    plt.title(
        "Complete dynamics vs derived effective closures"
    )

    plt.ylim(
        -20,
        20,
    )

    plt.legend()
    plt.tight_layout()

    fig.savefig(
        RESULTS_DIR
        / "006c_underlying_vs_closure.png",
        dpi=160,
    )

    plt.close(fig)

    # ---------------------------------------------------------------
    # Plot 2: closure fit
    # ---------------------------------------------------------------

    x_plot = np.linspace(
        -2.0,
        2.0,
        800,
    )

    fig = plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        x_plot,
        g(x_plot),
        label="complete g(x) = x + x³ - x⁵",
        linewidth=2,
    )

    for degree in CLOSURE_DEGREES:

        coefficients = model_store[
            (
                epsilon_plot,
                degree,
            )
        ]

        plt.plot(
            x_plot,
            evaluate_closure(
                x_plot,
                coefficients,
            ),
            label=f"degree {degree} fitted closure",
        )

    plt.axvline(
        TRAIN_LIMIT,
        linestyle="--",
        linewidth=1,
    )

    plt.axvline(
        -TRAIN_LIMIT,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel("x")
    plt.ylabel("dx/dt")
    plt.title(
        "The stabilising high-order structure lies outside the training regime"
    )

    plt.legend()
    plt.tight_layout()

    fig.savefig(
        RESULTS_DIR
        / "006c_closure_fit.png",
        dpi=160,
    )

    plt.close(fig)

    # ---------------------------------------------------------------
    # Plot 3: blow-up comparison
    # ---------------------------------------------------------------

    fig = plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        times_true,
        x_true,
        label="complete underlying system",
        linewidth=2,
    )

    coefficients = model_store[
        (
            epsilon_plot,
            3,
        )
    ]

    (
        times_cubic,
        x_cubic,
        failure_cubic,
    ) = integrate_closure(
        TEST_X0,
        coefficients,
        DT,
        T_FINAL,
    )

    plt.plot(
        times_cubic,
        x_cubic,
        label="derived cubic closure",
        linewidth=2,
    )

    plt.xlabel("time")
    plt.ylabel("x(t)")
    plt.title(
        "A bounded underlying system and a potentially singular closure"
    )

    plt.ylim(
        -20,
        20,
    )

    plt.legend()
    plt.tight_layout()

    fig.savefig(
        RESULTS_DIR
        / "006c_blowup_comparison.png",
        dpi=160,
    )

    plt.close(fig)

    print(
        "Results written to:"
    )

    print(
        f"  {csv_path}"
    )

    print(
        f"  {summary_path}"
    )

    print(
        "  results/006c_underlying_vs_closure.png"
    )

    print(
        "  results/006c_closure_fit.png"
    )

    print(
        "  results/006c_blowup_comparison.png"
    )


if __name__ == "__main__":
    main()
