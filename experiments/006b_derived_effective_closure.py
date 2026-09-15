"""
Experiment 006b: Derived Effective Closure
==========================================

Question
--------
Can a reduced description derived only from observed variables become
pathological when hidden variables are discarded?

Experiment 006 demonstrated that a singular effective law can coexist
with completely regular microscopic dynamics, but the singularity was
inserted by hand.

Experiment 006b removes that artificial singularity.

We use the regular microscopic system

    dx/dt = -y
    dy/dt =  x

and observe only x.

The exact reduced equation is

    d2x/dt2 + x = 0.

However, we deliberately attempt to construct a first-order autonomous
closure

    dx/dt = F(x)

using only the observed variable x.

Because y has been discarded, x alone is not dynamically closed.

The experiment:

1. generates trajectories from the full system;
2. estimates dx/dt from x(t);
3. fits polynomial first-order closures of increasing complexity;
4. measures their prediction error;
5. tests whether the fitted closure remains bounded;
6. compares the result with the exact second-order reduction.

The experiment does NOT assume a singular term.

The purpose is to determine what kind of pathology is actually produced
when a dynamically incomplete representation is forced into an autonomous
first-order form.

Outputs
-------
results/006b_closure_fits.csv
results/006b_closure_summary.txt
results/006b_observed_phase_relation.png
results/006b_closure_predictions.png
results/006b_long_term_prediction.png
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = Path("results")

DT = 0.002
T_FINAL = 40.0

TRAIN_TIME = 10.0

# Initial x values used to generate independent trajectories.
X0_VALUES = np.array([
    -0.9,
    -0.6,
    -0.3,
    0.0,
    0.3,
    0.6,
    0.9,
])

# Hidden-state values.
Y0_VALUES = np.array([
    -0.9,
    0.0,
    0.9,
])

POLYNOMIAL_DEGREES = np.array([
    1,
    2,
    3,
    5,
    7,
    9,
])

# Small ridge parameter for numerical stability.
RIDGE = 1.0e-10


# ---------------------------------------------------------------------------
# Microscopic dynamics
# ---------------------------------------------------------------------------

def micro_rhs(
    x: float,
    y: float,
) -> tuple[float, float]:

    dxdt = -y
    dydt = x

    return dxdt, dydt


def rk4_step(
    x: float,
    y: float,
    dt: float,
) -> tuple[float, float]:

    k1x, k1y = micro_rhs(x, y)

    k2x, k2y = micro_rhs(
        x + 0.5 * dt * k1x,
        y + 0.5 * dt * k1y,
    )

    k3x, k3y = micro_rhs(
        x + 0.5 * dt * k2x,
        y + 0.5 * dt * k2y,
    )

    k4x, k4y = micro_rhs(
        x + dt * k3x,
        y + dt * k3y,
    )

    x_new = x + (dt / 6.0) * (
        k1x
        + 2.0 * k2x
        + 2.0 * k3x
        + k4x
    )

    y_new = y + (dt / 6.0) * (
        k1y
        + 2.0 * k2y
        + 2.0 * k3y
        + k4y
    )

    return x_new, y_new


def integrate_micro(
    x0: float,
    y0: float,
    dt: float,
    t_final: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

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

        x[i + 1], y[i + 1] = rk4_step(
            x[i],
            y[i],
            dt,
        )

    return times, x, y


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_x(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:

    del y

    return np.asarray(x)


# ---------------------------------------------------------------------------
# Derivative estimation
# ---------------------------------------------------------------------------

def estimate_derivative(
    x: np.ndarray,
    dt: float,
) -> np.ndarray:

    return np.gradient(
        x,
        dt,
    )


# ---------------------------------------------------------------------------
# Polynomial closure
# ---------------------------------------------------------------------------

def polynomial_design_matrix(
    x: np.ndarray,
    degree: int,
) -> np.ndarray:

    return np.column_stack(
        [
            x ** power
            for power in range(
                degree + 1
            )
        ]
    )


def fit_polynomial_closure(
    x: np.ndarray,
    dxdt: np.ndarray,
    degree: int,
) -> np.ndarray:
    """
    Fit

        dx/dt = c0 + c1*x + ... + cn*x^n

    using ridge-regularised least squares.
    """

    A = polynomial_design_matrix(
        x,
        degree,
    )

    ATA = A.T @ A

    regularisation = RIDGE * np.eye(
        ATA.shape[0]
    )

    coefficients = np.linalg.solve(
        ATA + regularisation,
        A.T @ dxdt,
    )

    return coefficients


def evaluate_polynomial(
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
# Closure integration
# ---------------------------------------------------------------------------

def closure_rhs(
    x: float,
    coefficients: np.ndarray,
) -> float:

    value = 0.0

    for power, coefficient in enumerate(
        coefficients
    ):

        value += (
            coefficient
            * x**power
        )

    return float(value)


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
    safety_limit: float = 1.0e6,
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
        dtype=float,
    )

    x[0] = x0

    failure_time = None

    for i in range(n_steps):

        if not np.isfinite(
            x[i]
        ):

            failure_time = times[i]
            break

        if abs(x[i]) > safety_limit:

            failure_time = times[i]
            break

        x_next = rk4_closure_step(
            x[i],
            dt,
            coefficients,
        )

        x[i + 1] = x_next

        if (
            not np.isfinite(x_next)
            or abs(x_next) > safety_limit
        ):

            failure_time = times[i + 1]
            break

    return (
        times,
        x,
        failure_time,
    )


# ---------------------------------------------------------------------------
# Exact reduced dynamics
# ---------------------------------------------------------------------------

def exact_reduced_rhs(
    x: float,
    velocity: float,
) -> tuple[float, float]:
    """
    Exact second-order reduction:

        dx/dt = v
        dv/dt = -x

    where v = dx/dt.

    This is mathematically equivalent to the full (x,y) system after
    identifying

        v = -y.
    """

    dxdt = velocity
    dvdt = -x

    return dxdt, dvdt


def integrate_exact_reduced(
    x0: float,
    y0: float,
    dt: float,
    t_final: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    velocity0 = -y0

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

    velocity = np.empty(
        n_steps + 1
    )

    x[0] = x0
    velocity[0] = velocity0

    def rhs(
        xx: float,
        vv: float,
    ) -> tuple[float, float]:

        return exact_reduced_rhs(
            xx,
            vv,
        )

    for i in range(n_steps):

        k1x, k1v = rhs(
            x[i],
            velocity[i],
        )

        k2x, k2v = rhs(
            x[i] + 0.5 * dt * k1x,
            velocity[i] + 0.5 * dt * k1v,
        )

        k3x, k3v = rhs(
            x[i] + 0.5 * dt * k2x,
            velocity[i] + 0.5 * dt * k2v,
        )

        k4x, k4v = rhs(
            x[i] + dt * k3x,
            velocity[i] + dt * k3v,
        )

        x[i + 1] = (
            x[i]
            + (dt / 6.0)
            * (
                k1x
                + 2.0 * k2x
                + 2.0 * k3x
                + k4x
            )
        )

        velocity[i + 1] = (
            velocity[i]
            + (dt / 6.0)
            * (
                k1v
                + 2.0 * k2v
                + 2.0 * k3v
                + k4v
            )
        )

    return (
        times,
        x,
        velocity,
    )


# ---------------------------------------------------------------------------
# Training dataset
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    x: np.ndarray
    dxdt: np.ndarray


def build_training_dataset() -> Dataset:

    x_values = []
    dxdt_values = []

    train_steps = int(
        round(TRAIN_TIME / DT)
    )

    for x0 in X0_VALUES:

        for y0 in Y0_VALUES:

            radius = np.sqrt(
                x0**2 + y0**2
            )

            if radius > 1.0:
                scale = 1.0 / radius
                x_initial = x0 * scale
                y_initial = y0 * scale
            else:
                x_initial = x0
                y_initial = y0

            times, x, y = integrate_micro(
                x_initial,
                y_initial,
                DT,
                TRAIN_TIME,
            )

            projected = project_x(
                x,
                y,
            )

            derivative = estimate_derivative(
                projected,
                DT,
            )

            x_values.append(
                projected[: train_steps + 1]
            )

            dxdt_values.append(
                derivative[: train_steps + 1]
            )

    return Dataset(
        x=np.concatenate(
            x_values
        ),
        dxdt=np.concatenate(
            dxdt_values
        ),
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


def evaluate_fit(
    dataset: Dataset,
    coefficients: np.ndarray,
) -> tuple[float, float]:

    predicted = evaluate_polynomial(
        dataset.x,
        coefficients,
    )

    error = (
        predicted
        - dataset.dxdt
    )

    return (
        rms(error),
        float(
            np.max(
                np.abs(error)
            )
        ),
    )


def closure_variance(
    dataset: Dataset,
    n_bins: int = 80,
) -> tuple[float, float]:

    x = dataset.x
    dxdt = dataset.dxdt

    edges = np.linspace(
        np.min(x),
        np.max(x),
        n_bins + 1,
    )

    indices = np.digitize(
        x,
        edges,
    ) - 1

    standard_deviations = []

    for i in range(n_bins):

        mask = indices == i

        if np.sum(mask) >= 5:

            standard_deviations.append(
                np.std(
                    dxdt[mask]
                )
            )

    if not standard_deviations:

        return 0.0, 0.0

    return (
        float(
            np.mean(
                standard_deviations
            )
        ),
        float(
            np.max(
                standard_deviations
            )
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(
        "Experiment 006b: Derived Effective Closure"
    )
    print("=" * 72)
    print()

    print(
        "Generating projected training data..."
    )

    dataset = build_training_dataset()

    print(
        f"Training samples: {len(dataset.x)}"
    )

    print()

    mean_closure_std, max_closure_std = (
        closure_variance(
            dataset
        )
    )

    print(
        "Hidden-state closure diagnostic:"
    )

    print(
        f"  mean within-x derivative std = "
        f"{mean_closure_std:.6f}"
    )

    print(
        f"  max within-x derivative std  = "
        f"{max_closure_std:.6f}"
    )

    print()

    # ---------------------------------------------------------------
    # Fit polynomial closures
    # ---------------------------------------------------------------

    fit_results = []

    fitted_models = {}

    for degree in POLYNOMIAL_DEGREES:

        coefficients = fit_polynomial_closure(
            dataset.x,
            dataset.dxdt,
            int(degree),
        )

        fit_rms, fit_max = evaluate_fit(
            dataset,
            coefficients,
        )

        fitted_models[int(degree)] = (
            coefficients
        )

        fit_results.append(
            (
                int(degree),
                fit_rms,
                fit_max,
            )
        )

        print(
            f"degree={degree:2d}  "
            f"training_RMS={fit_rms:.8f}  "
            f"training_max={fit_max:.8f}"
        )

    print()

    # ---------------------------------------------------------------
    # Long-term prediction tests
    # ---------------------------------------------------------------

    print(
        "Long-term closure prediction:"
    )

    prediction_results = []

    test_x0 = 0.2
    test_y0 = 0.8

    times_true, x_true, y_true = (
        integrate_micro(
            test_x0,
            test_y0,
            DT,
            T_FINAL,
        )
    )

    for degree in POLYNOMIAL_DEGREES:

        coefficients = fitted_models[
            int(degree)
        ]

        times_model, x_model, failure = (
            integrate_closure(
                test_x0,
                coefficients,
                DT,
                T_FINAL,
            )
        )

        valid = np.isfinite(
            x_model
        )

        if np.any(valid):

            common = (
                valid
                & np.isfinite(x_true)
            )

            prediction_rms = rms(
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

            max_model_abs = float(
                np.max(
                    np.abs(
                        x_model[common]
                    )
                )
            )

        else:

            prediction_rms = np.inf
            max_prediction_error = np.inf
            max_model_abs = np.inf

        prediction_results.append(
            (
                int(degree),
                prediction_rms,
                max_prediction_error,
                max_model_abs,
                failure,
            )
        )

        failure_text = (
            "none"
            if failure is None
            else f"{failure:.6f}"
        )

        print(
            f"degree={degree:2d}  "
            f"prediction_RMS={prediction_rms:.8f}  "
            f"prediction_max={max_prediction_error:.8f}  "
            f"max|x|={max_model_abs:.8f}  "
            f"failure={failure_text}"
        )

    print()

    # ---------------------------------------------------------------
    # Exact reduced model
    # ---------------------------------------------------------------

    times_exact, x_exact, velocity_exact = (
        integrate_exact_reduced(
            test_x0,
            test_y0,
            DT,
            T_FINAL,
        )
    )

    exact_error = rms(
        x_exact - x_true
    )

    print(
        "Exact second-order reduction:"
    )

    print(
        f"  RMS error against microscopic x = "
        f"{exact_error:.8e}"
    )

    print(
        f"  maximum |x| = "
        f"{np.max(np.abs(x_exact)):.6f}"
    )

    print()

    # ---------------------------------------------------------------
    # CSV
    # ---------------------------------------------------------------

    csv_path = (
        RESULTS_DIR
        / "006b_closure_fits.csv"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as handle:

        writer = csv.writer(
            handle
        )

        writer.writerow(
            [
                "degree",
                "training_rms",
                "training_max",
                "prediction_rms",
                "prediction_max",
                "prediction_max_abs_x",
                "failure_time",
            ]
        )

        for (
            degree,
            fit_rms,
            fit_max,
        ) in fit_results:

            prediction = next(
                item
                for item in prediction_results
                if item[0] == degree
            )

            (
                _,
                prediction_rms,
                prediction_max,
                max_abs_x,
                failure,
            ) = prediction

            writer.writerow(
                [
                    degree,
                    fit_rms,
                    fit_max,
                    prediction_rms,
                    prediction_max,
                    max_abs_x,
                    failure,
                ]
            )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    summary_path = (
        RESULTS_DIR
        / "006b_closure_summary.txt"
    )

    with summary_path.open(
        "w"
    ) as handle:

        handle.write(
            "Experiment 006b: Derived Effective Closure\n"
        )

        handle.write(
            "=" * 72 + "\n\n"
        )

        handle.write(
            "Question:\n"
        )

        handle.write(
            "Can a reduced description derived only from observed "
            "variables become pathological when hidden variables "
            "are discarded?\n\n"
        )

        handle.write(
            "Microscopic dynamics:\n"
            "    dx/dt = -y\n"
            "    dy/dt =  x\n\n"
        )

        handle.write(
            "Projection:\n"
            "    C(x,y) = x\n\n"
        )

        handle.write(
            "Exact reduced equation:\n"
            "    d2x/dt2 + x = 0\n\n"
        )

        handle.write(
            "Attempted effective closure:\n"
            "    dx/dt = F(x)\n\n"
        )

        handle.write(
            "Closure diagnostic:\n"
        )

        handle.write(
            f"    mean within-x derivative std = "
            f"{mean_closure_std:.8f}\n"
        )

        handle.write(
            f"    max within-x derivative std = "
            f"{max_closure_std:.8f}\n\n"
        )

        handle.write(
            "Polynomial fit results:\n"
        )

        for (
            degree,
            fit_rms,
            fit_max,
        ) in fit_results:

            handle.write(
                f"    degree={degree}: "
                f"training RMS={fit_rms:.8f}, "
                f"training max={fit_max:.8f}\n"
            )

        handle.write(
            "\nLong-term prediction results:\n"
        )

        for result in prediction_results:

            (
                degree,
                prediction_rms,
                prediction_max,
                max_abs_x,
                failure,
            ) = result

            handle.write(
                f"    degree={degree}: "
                f"RMS={prediction_rms:.8f}, "
                f"max error={prediction_max:.8f}, "
                f"max|x|={max_abs_x:.8f}, "
                f"failure={failure}\n"
            )

        handle.write(
            "\nExact second-order reduction:\n"
        )

        handle.write(
            f"    RMS error against microscopic x = "
            f"{exact_error:.8e}\n"
        )

        handle.write(
            "\nInterpretation:\n"
        )

        handle.write(
            "The full microscopic system is regular and bounded.\n\n"
        )

        handle.write(
            "Discarding y makes x dynamically non-closed because "
            "the same x can occur with different hidden y and therefore "
            "different dx/dt.\n\n"
        )

        handle.write(
            "The exact reduced description does not become singular. "
            "Instead, eliminating y produces a second-order equation.\n\n"
        )

        handle.write(
            "Therefore projection alone does not imply singularity.\n\n"
        )

        handle.write(
            "The more interesting question is whether insisting on an "
            "insufficient first-order closure can produce increasingly "
            "pathological behaviour as the underlying dynamics become "
            "more nonlinear or as more hidden variables are discarded.\n"
        )

    # ---------------------------------------------------------------
    # Plot: observed phase relation
    # ---------------------------------------------------------------

    fig = plt.figure(
        figsize=(9, 6)
    )

    plt.scatter(
        dataset.x,
        dataset.dxdt,
        s=3,
        alpha=0.15,
    )

    x_plot = np.linspace(
        np.min(dataset.x),
        np.max(dataset.x),
        500,
    )

    for degree in (
        1,
        3,
        7,
    ):

        coefficients = fitted_models[
            degree
        ]

        plt.plot(
            x_plot,
            evaluate_polynomial(
                x_plot,
                coefficients,
            ),
            label=f"degree {degree}",
        )

    plt.xlabel("observed x")
    plt.ylabel("dx/dt")
    plt.title(
        "Projected dynamics: x alone does not define a unique derivative"
    )
    plt.legend()
    plt.tight_layout()

    fig.savefig(
        RESULTS_DIR
        / "006b_observed_phase_relation.png",
        dpi=160,
    )

    plt.close(fig)

    # ---------------------------------------------------------------
    # Plot: closure predictions
    # ---------------------------------------------------------------

    fig = plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        times_true,
        x_true,
        label="microscopic truth",
        linewidth=2,
    )

    for degree in (
        1,
        3,
        5,
        7,
    ):

        coefficients = fitted_models[
            degree
        ]

        times_model, x_model, failure = (
            integrate_closure(
                test_x0,
                coefficients,
                DT,
                T_FINAL,
            )
        )

        plt.plot(
            times_model,
            x_model,
            label=f"degree {degree}",
        )

    plt.xlabel("time")
    plt.ylabel("x(t)")
    plt.title(
        "Derived first-order closures vs microscopic dynamics"
    )
    plt.legend()
    plt.tight_layout()

    fig.savefig(
        RESULTS_DIR
        / "006b_closure_predictions.png",
        dpi=160,
    )

    plt.close(fig)

    # ---------------------------------------------------------------
    # Plot: exact reduced model and long-term behaviour
    # ---------------------------------------------------------------

    fig = plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        times_true,
        x_true,
        label="microscopic truth",
        linewidth=2,
    )

    plt.plot(
        times_exact,
        x_exact,
        "--",
        label="exact second-order reduction",
    )

    # Plot the most complex closure.
    coefficients = fitted_models[9]

    times_model, x_model, failure = (
        integrate_closure(
            test_x0,
            coefficients,
            DT,
            T_FINAL,
        )
    )

    plt.plot(
        times_model,
        x_model,
        label="degree 9 first-order closure",
    )

    plt.xlabel("time")
    plt.ylabel("x(t)")
    plt.title(
        "Exact reduction vs insufficient first-order closure"
    )
    plt.legend()
    plt.tight_layout()

    fig.savefig(
        RESULTS_DIR
        / "006b_long_term_prediction.png",
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
        "  results/006b_observed_phase_relation.png"
    )

    print(
        "  results/006b_closure_predictions.png"
    )

    print(
        "  results/006b_long_term_prediction.png"
    )


if __name__ == "__main__":
    main()
