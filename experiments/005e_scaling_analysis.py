"""
Experiment 005e: Scaling analysis of Experiment 005d.

Purpose
-------
Analyze the existing 005d results without running new simulations.

Experiment 005d measured

    Delta = Delta(L, tau)

with L and tau varied independently.

This experiment asks:

1. How strongly does Delta depend on tau?
2. How strongly does Delta depend on L?
3. Does epsilon = tau/L explain the data?
4. Are there approximately scale-independent plateaus?
5. Where is sensitivity to L largest?
6. Can the observed behaviour be summarized by a simple empirical
   scaling relationship?

This is descriptive analysis only. No new physical model is introduced.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INPUT_FILE = Path(
    "results/005d_scale_vs_relaxation/results.csv"
)

OUTPUT_DIR = Path(
    "results/005e_scaling_analysis"
)


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------


def load_results(path):
    """
    Load the 005d CSV using only the Python standard library.

    Returns a list of dictionaries with numerical values.
    """

    import csv

    if not path.exists():
        raise FileNotFoundError(
            f"Could not find 005d results at:\n{path}"
        )

    with open(
        path,
        "r",
        newline="",
    ) as handle:

        reader = csv.DictReader(
            handle
        )

        rows = []

        for row in reader:

            rows.append(
                {
                    key: float(value)
                    for key, value
                    in row.items()
                }
            )

    if not rows:
        raise RuntimeError(
            "005d results file contains no data."
        )

    return rows


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def unique_sorted(
    rows,
    key,
):
    """
    Return sorted unique values for a field.
    """

    return sorted(
        {
            row[key]
            for row in rows
        }
    )


def nearest_row(
    rows,
    field,
    value,
):
    """
    Return row whose field is closest to value.
    """

    return min(
        rows,
        key=lambda row:
        abs(row[field] - value),
    )


def rows_at_tau(
    rows,
    tau,
):
    """
    Select rows at approximately fixed tau.
    """

    return [
        row
        for row in rows
        if np.isclose(
            row["tau"],
            tau,
            rtol=1e-10,
            atol=1e-12,
        )
    ]


def rows_at_L(
    rows,
    L,
):
    """
    Select rows at approximately fixed L.
    """

    return [
        row
        for row in rows
        if np.isclose(
            row["L"],
            L,
            rtol=1e-10,
            atol=1e-12,
        )
    ]


# ---------------------------------------------------------------------
# Fixed-tau scale sensitivity
# ---------------------------------------------------------------------


def calculate_scale_sensitivity(
    rows,
):
    """
    Calculate finite differences in Delta with respect to log(L).

    The quantity

        dDelta / dlog(L)

    measures how strongly discrepancy changes under multiplicative
    changes in physical scale.

    Values near zero indicate a scale-insensitive region.
    """

    records = []

    tau_values = unique_sorted(
        rows,
        "tau",
    )

    for tau in tau_values:

        subset = sorted(
            rows_at_tau(
                rows,
                tau,
            ),
            key=lambda row:
            row["L"],
        )

        for left, right in zip(
            subset[:-1],
            subset[1:],
        ):

            delta_left = (
                left["final_combined"]
            )

            delta_right = (
                right["final_combined"]
            )

            log_L_left = np.log(
                left["L"]
            )

            log_L_right = np.log(
                right["L"]
            )

            sensitivity = (
                delta_right
                - delta_left
            ) / (
                log_L_right
                - log_L_left
            )

            records.append(
                {
                    "tau": tau,
                    "L_left": left["L"],
                    "L_right": right["L"],
                    "delta_left": delta_left,
                    "delta_right": delta_right,
                    "dDelta_dlogL": sensitivity,
                }
            )

    return records


# ---------------------------------------------------------------------
# Fixed-L relaxation sensitivity
# ---------------------------------------------------------------------


def calculate_relaxation_sensitivity(
    rows,
):
    """
    Calculate finite differences in Delta with respect to log(tau).

    This provides a directly comparable measure of sensitivity to
    relaxation time.
    """

    records = []

    L_values = unique_sorted(
        rows,
        "L",
    )

    for L in L_values:

        subset = sorted(
            rows_at_L(
                rows,
                L,
            ),
            key=lambda row:
            row["tau"],
        )

        for left, right in zip(
            subset[:-1],
            subset[1:],
        ):

            delta_left = (
                left["final_combined"]
            )

            delta_right = (
                right["final_combined"]
            )

            log_tau_left = np.log(
                left["tau"]
            )

            log_tau_right = np.log(
                right["tau"]
            )

            sensitivity = (
                delta_right
                - delta_left
            ) / (
                log_tau_right
                - log_tau_left
            )

            records.append(
                {
                    "L": L,
                    "tau_left": left["tau"],
                    "tau_right": right["tau"],
                    "delta_left": delta_left,
                    "delta_right": delta_right,
                    "dDelta_dlogtau": sensitivity,
                }
            )

    return records


# ---------------------------------------------------------------------
# Equal-epsilon comparisons
# ---------------------------------------------------------------------


def equal_epsilon_groups(
    rows,
    tolerance=1e-8,
):
    """
    Group cases with approximately equal epsilon.

    In the current 005d grid, some epsilon values occur at multiple
    physical scales.
    """

    groups = {}

    for row in rows:

        epsilon = row["epsilon"]

        matched_key = None

        for key in groups:

            if abs(
                epsilon - key
            ) <= tolerance:

                matched_key = key
                break

        if matched_key is None:

            groups[epsilon] = [
                row
            ]

        else:

            groups[
                matched_key
            ].append(row)

    return {
        key: value
        for key, value in groups.items()
        if len(value) > 1
    }


# ---------------------------------------------------------------------
# Equal-epsilon spread
# ---------------------------------------------------------------------


def calculate_epsilon_spread(
    rows,
):
    """
    Measure how much Delta varies among cases sharing the same epsilon.

    If epsilon were sufficient to determine closure error, these spreads
    would be small.
    """

    records = []

    groups = equal_epsilon_groups(
        rows
    )

    for epsilon, group in sorted(
        groups.items()
    ):

        deltas = np.array(
            [
                row["final_combined"]
                for row in group
            ]
        )

        records.append(
            {
                "epsilon": epsilon,
                "n": len(group),
                "minimum_delta": float(
                    np.min(deltas)
                ),
                "maximum_delta": float(
                    np.max(deltas)
                ),
                "mean_delta": float(
                    np.mean(deltas)
                ),
                "spread": float(
                    np.max(deltas)
                    - np.min(deltas)
                ),
                "ratio_max_min": float(
                    np.max(deltas)
                    / max(
                        np.min(deltas),
                        1e-15,
                    )
                ),
            }
        )

    return records


# ---------------------------------------------------------------------
# Scale ratios
# ---------------------------------------------------------------------


def calculate_scale_ratios(
    rows,
):
    """
    Compare largest-scale and smallest-scale discrepancy at fixed tau.
    """

    records = []

    tau_values = unique_sorted(
        rows,
        "tau",
    )

    for tau in tau_values:

        subset = sorted(
            rows_at_tau(
                rows,
                tau,
            ),
            key=lambda row:
            row["L"],
        )

        smallest = subset[0]
        largest = subset[-1]

        delta_small = (
            smallest["final_combined"]
        )

        delta_large = (
            largest["final_combined"]
        )

        records.append(
            {
                "tau": tau,
                "L_small": smallest["L"],
                "L_large": largest["L"],
                "delta_small": delta_small,
                "delta_large": delta_large,
                "large_to_small_ratio": (
                    delta_large
                    / max(
                        delta_small,
                        1e-15,
                    )
                ),
                "absolute_difference": (
                    delta_large
                    - delta_small
                ),
            }
        )

    return records


# ---------------------------------------------------------------------
# Plateau analysis
# ---------------------------------------------------------------------


def calculate_plateau_metrics(
    rows,
):
    """
    Compare the last two scale values at fixed tau.

    A small relative difference indicates that Delta has become
    approximately insensitive to further reduction in L.
    """

    records = []

    tau_values = unique_sorted(
        rows,
        "tau",
    )

    for tau in tau_values:

        subset = sorted(
            rows_at_tau(
                rows,
                tau,
            ),
            key=lambda row:
            row["L"],
        )

        if len(subset) < 2:
            continue

        previous = subset[-2]
        smallest = subset[-1]

        delta_previous = (
            previous["final_combined"]
        )

        delta_smallest = (
            smallest["final_combined"]
        )

        absolute_change = (
            delta_smallest
            - delta_previous
        )

        relative_change = (
            absolute_change
            / max(
                abs(delta_previous),
                1e-15,
            )
        )

        records.append(
            {
                "tau": tau,
                "L_previous": previous["L"],
                "L_smallest": smallest["L"],
                "delta_previous": delta_previous,
                "delta_smallest": delta_smallest,
                "absolute_change": absolute_change,
                "relative_change": relative_change,
            }
        )

    return records


# ---------------------------------------------------------------------
# Power-law fit
# ---------------------------------------------------------------------


def fit_power_law(
    x,
    y,
):
    """
    Fit

        y = A * x^p

    using log-log least squares.

    Returns A, p, R^2.

    This is purely descriptive; it is not assumed to be a physical law.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    valid = (
        (x > 0)
        & (y > 0)
    )

    x = x[valid]
    y = y[valid]

    if len(x) < 2:
        return (
            np.nan,
            np.nan,
            np.nan,
        )

    log_x = np.log(x)
    log_y = np.log(y)

    slope, intercept = np.polyfit(
        log_x,
        log_y,
        1,
    )

    prediction = (
        intercept
        + slope * log_x
    )

    ss_res = np.sum(
        (log_y - prediction) ** 2
    )

    ss_tot = np.sum(
        (
            log_y
            - np.mean(log_y)
        ) ** 2
    )

    if ss_tot > 0.0:
        r_squared = (
            1.0
            - ss_res / ss_tot
        )
    else:
        r_squared = np.nan

    A = np.exp(
        intercept
    )

    return (
        A,
        slope,
        r_squared,
    )


def calculate_power_laws(
    rows,
):
    """
    Fit Delta ~ tau^p at fixed L.

    Also fit Delta ~ L^q at fixed tau.
    """

    records = []

    # Delta versus tau at fixed L.
    for L in unique_sorted(
        rows,
        "L",
    ):

        subset = sorted(
            rows_at_L(
                rows,
                L,
            ),
            key=lambda row:
            row["tau"],
        )

        tau_values = [
            row["tau"]
            for row in subset
        ]

        delta_values = [
            row["final_combined"]
            for row in subset
        ]

        A, exponent, r2 = (
            fit_power_law(
                tau_values,
                delta_values,
            )
        )

        records.append(
            {
                "relationship": (
                    "Delta_vs_tau"
                ),
                "fixed_value": L,
                "A": A,
                "exponent": exponent,
                "R2_log": r2,
            }
        )

    # Delta versus L at fixed tau.
    for tau in unique_sorted(
        rows,
        "tau",
    ):

        subset = sorted(
            rows_at_tau(
                rows,
                tau,
            ),
            key=lambda row:
            row["L"],
        )

        L_values = [
            row["L"]
            for row in subset
        ]

        delta_values = [
            row["final_combined"]
            for row in subset
        ]

        A, exponent, r2 = (
            fit_power_law(
                L_values,
                delta_values,
            )
        )

        records.append(
            {
                "relationship": (
                    "Delta_vs_L"
                ),
                "fixed_value": tau,
                "A": A,
                "exponent": exponent,
                "R2_log": r2,
            }
        )

    return records


# ---------------------------------------------------------------------
# Plot: sensitivity comparison
# ---------------------------------------------------------------------


def plot_sensitivity_comparison(
    scale_sensitivity,
    relaxation_sensitivity,
    output_path,
):
    """
    Compare absolute sensitivity to log(L) and log(tau).

    This is not a formal derivative estimate; it is a finite-difference
    diagnostic over the sampled parameter grid.
    """

    tau_values = sorted(
        {
            record["tau"]
            for record in scale_sensitivity
        }
    )

    scale_magnitudes = []

    relaxation_magnitudes = []

    for tau in tau_values:

        scale_records = [
            record
            for record in scale_sensitivity
            if np.isclose(
                record["tau"],
                tau,
            )
        ]

        scale_magnitude = np.mean(
            [
                abs(
                    record[
                        "dDelta_dlogL"
                    ]
                )
                for record
                in scale_records
            ]
        )

        scale_magnitudes.append(
            scale_magnitude
        )

        relaxation_records = (
            relaxation_sensitivity
        )

        relaxation_magnitude = np.mean(
            [
                abs(
                    record[
                        "dDelta_dlogtau"
                    ]
                )
                for record
                in relaxation_records
            ]
        )

        relaxation_magnitudes.append(
            relaxation_magnitude
        )

    plt.figure()

    plt.plot(
        tau_values,
        scale_magnitudes,
        marker="o",
        label="scale sensitivity",
    )

    plt.plot(
        tau_values,
        relaxation_magnitudes,
        marker="o",
        label="relaxation sensitivity",
    )

    plt.xlabel(
        "tau"
    )

    plt.ylabel(
        "Mean absolute sensitivity"
    )

    plt.title(
        "005e: Sensitivity to scale vs relaxation"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()


# ---------------------------------------------------------------------
# Plot: epsilon spread
# ---------------------------------------------------------------------


def plot_epsilon_spread(
    epsilon_spread,
    output_path,
):
    """
    Plot discrepancy spread among equal-epsilon cases.
    """

    if not epsilon_spread:
        return

    epsilon = [
        record["epsilon"]
        for record in epsilon_spread
    ]

    spread = [
        record["spread"]
        for record in epsilon_spread
    ]

    ratio = [
        record["ratio_max_min"]
        for record in epsilon_spread
    ]

    plt.figure()

    plt.plot(
        epsilon,
        spread,
        marker="o",
    )

    plt.xlabel(
        "epsilon = tau / L"
    )

    plt.ylabel(
        "Delta spread at fixed epsilon"
    )

    plt.title(
        "005e: Failure of epsilon-only collapse"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=160,
    )

    plt.close()

    # Ratio plot
    plt.figure()

    plt.plot(
        epsilon,
        ratio,
        marker="o",
    )

    plt.xlabel(
        "epsilon = tau / L"
    )

    plt.ylabel(
        "max Delta / min Delta"
    )

    plt.title(
        "005e: Equal-epsilon discrepancy ratio"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path.parent
        / "epsilon_ratio.png",
        dpi=160,
    )

    plt.close()


# ---------------------------------------------------------------------
# Save text report
# ---------------------------------------------------------------------


def write_report(
    rows,
    scale_sensitivity,
    relaxation_sensitivity,
    epsilon_spread,
    scale_ratios,
    plateau_metrics,
    power_laws,
    output_path,
):
    """
    Write a complete numerical report.
    """

    with open(
        output_path,
        "w",
    ) as handle:

        handle.write(
            "Experiment 005e: Scaling analysis\n"
        )

        handle.write(
            "====================================\n\n"
        )

        handle.write(
            "This report analyzes Experiment 005d "
            "without running new simulations.\n\n"
        )

        # ----------------------------------------------------------
        # Basic data
        # ----------------------------------------------------------

        handle.write(
            "1. DATASET\n"
        )

        handle.write(
            "----------\n"
        )

        handle.write(
            f"Cases: {len(rows)}\n"
        )

        handle.write(
            "Spatial scales: "
            + ", ".join(
                f"{value:.8f}"
                for value
                in unique_sorted(
                    rows,
                    "L",
                )
            )
            + "\n"
        )

        handle.write(
            "Relaxation times: "
            + ", ".join(
                f"{value:.5f}"
                for value
                in unique_sorted(
                    rows,
                    "tau",
                )
            )
            + "\n\n"
        )

        # ----------------------------------------------------------
        # Scale sensitivity
        # ----------------------------------------------------------

        handle.write(
            "2. SCALE SENSITIVITY\n"
        )

        handle.write(
            "--------------------\n"
        )

        for record in scale_sensitivity:

            handle.write(
                f"tau={record['tau']:.5f} "
                f"L={record['L_left']:.6f}"
                f"->{record['L_right']:.6f} "
                f"dDelta/dlogL="
                f"{record['dDelta_dlogL']:.8f}\n"
            )

        handle.write(
            "\n"
        )

        # ----------------------------------------------------------
        # Relaxation sensitivity
        # ----------------------------------------------------------

        handle.write(
            "3. RELAXATION SENSITIVITY\n"
        )

        handle.write(
            "-------------------------\n"
        )

        for record in relaxation_sensitivity:

            handle.write(
                f"L={record['L']:.6f} "
                f"tau={record['tau_left']:.5f}"
                f"->{record['tau_right']:.5f} "
                f"dDelta/dlogtau="
                f"{record['dDelta_dlogtau']:.8f}\n"
            )

        handle.write(
            "\n"
        )

        # ----------------------------------------------------------
        # Equal epsilon
        # ----------------------------------------------------------

        handle.write(
            "4. EQUAL-EPSILON COMPARISONS\n"
        )

        handle.write(
            "-----------------------------\n"
        )

        for record in epsilon_spread:

            handle.write(
                f"epsilon={record['epsilon']:.8f} "
                f"n={record['n']} "
                f"min={record['minimum_delta']:.8f} "
                f"max={record['maximum_delta']:.8f} "
                f"spread={record['spread']:.8f} "
                f"ratio={record['ratio_max_min']:.4f}\n"
            )

        handle.write(
            "\n"
        )

        # ----------------------------------------------------------
        # Scale ratios
        # ----------------------------------------------------------

        handle.write(
            "5. LARGE-SCALE / SMALL-SCALE RATIOS\n"
        )

        handle.write(
            "-----------------------------------\n"
        )

        for record in scale_ratios:

            handle.write(
                f"tau={record['tau']:.5f} "
                f"small={record['delta_small']:.8f} "
                f"large={record['delta_large']:.8f} "
                f"ratio="
                f"{record['large_to_small_ratio']:.4f} "
                f"difference="
                f"{record['absolute_difference']:.8f}\n"
            )

        handle.write(
            "\n"
        )

        # ----------------------------------------------------------
        # Plateau
        # ----------------------------------------------------------

        handle.write(
            "6. SMALL-SCALE PLATEAU TEST\n"
        )

        handle.write(
            "----------------------------\n"
        )

        for record in plateau_metrics:

            handle.write(
                f"tau={record['tau']:.5f} "
                f"Delta(previous)="
                f"{record['delta_previous']:.8f} "
                f"Delta(smallest)="
                f"{record['delta_smallest']:.8f} "
                f"relative_change="
                f"{record['relative_change']:.6f}\n"
            )

        handle.write(
            "\n"
        )

        # ----------------------------------------------------------
        # Power laws
        # ----------------------------------------------------------

        handle.write(
            "7. DESCRIPTIVE POWER-LAW FITS\n"
        )

        handle.write(
            "-----------------------------\n"
        )

        handle.write(
            "These fits are descriptive only.\n\n"
        )

        for record in power_laws:

            handle.write(
                f"{record['relationship']} "
                f"fixed={record['fixed_value']:.8f} "
                f"A={record['A']:.8e} "
                f"exponent={record['exponent']:.8f} "
                f"R2_log={record['R2_log']:.8f}\n"
            )

        handle.write(
            "\n"
        )

        # ----------------------------------------------------------
        # Interpretation
        # ----------------------------------------------------------

        handle.write(
            "8. INTERPRETATION GUIDE\n"
        )

        handle.write(
            "-----------------------\n"
        )

        handle.write(
            "The principal comparisons are:\n\n"
        )

        handle.write(
            "A. Strong tau dependence with weak L dependence\n"
        )

        handle.write(
            "   -> closure error is primarily controlled by "
            "relaxation.\n\n"
        )

        handle.write(
            "B. Strong L dependence at fixed tau\n"
        )

        handle.write(
            "   -> physical scale contributes independently "
            "of relaxation.\n\n"
        )

        handle.write(
            "C. Large equal-epsilon spread\n"
        )

        handle.write(
            "   -> epsilon=tau/L is insufficient as a complete "
            "predictor in this experiment.\n\n"
        )

        handle.write(
            "D. Small change between the two smallest L values\n"
        )

        handle.write(
            "   -> possible scale-insensitive plateau, but "
            "additional experiments would be required before "
            "calling it a physical boundary.\n\n"
        )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    """
    Run the analysis.
    """

    rows = load_results(
        INPUT_FILE
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scale_sensitivity = (
        calculate_scale_sensitivity(
            rows
        )
    )

    relaxation_sensitivity = (
        calculate_relaxation_sensitivity(
            rows
        )
    )

    epsilon_spread = (
        calculate_epsilon_spread(
            rows
        )
    )

    scale_ratios = (
        calculate_scale_ratios(
            rows
        )
    )

    plateau_metrics = (
        calculate_plateau_metrics(
            rows
        )
    )

    power_laws = (
        calculate_power_laws(
            rows
        )
    )

    write_report(
        rows,
        scale_sensitivity,
        relaxation_sensitivity,
        epsilon_spread,
        scale_ratios,
        plateau_metrics,
        power_laws,
        OUTPUT_DIR
        / "analysis.txt",
    )

    plot_sensitivity_comparison(
        scale_sensitivity,
        relaxation_sensitivity,
        OUTPUT_DIR
        / "sensitivity_comparison.png",
    )

    plot_epsilon_spread(
        epsilon_spread,
        OUTPUT_DIR
        / "epsilon_spread.png",
    )

    print()
    print(
        "Experiment 005e: Scaling analysis"
    )
    print(
        "================================="
    )

    print(
        f"Loaded {len(rows)} cases from:"
    )

    print(
        f"  {INPUT_FILE}"
    )

    print()
    print(
        "Analysis written to:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print()
    print(
        "Main file:"
    )

    print(
        f"  {OUTPUT_DIR / 'analysis.txt'}"
    )

    print()
    print(
        "Plots:"
    )

    print(
        f"  {OUTPUT_DIR / 'sensitivity_comparison.png'}"
    )

    print(
        f"  {OUTPUT_DIR / 'epsilon_spread.png'}"
    )

    print(
        f"  {OUTPUT_DIR / 'epsilon_ratio.png'}"
    )


if __name__ == "__main__":
    main()
