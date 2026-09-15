"""
Experiment 006d: Genuine Moment Hierarchy and Natural Closure

Question
--------
Can a regular microscopic dynamical system contain stabilising higher-order
structure that is absent from a low-order effective representation?

Unlike Experiment 006c, this experiment does NOT fit a polynomial closure
to trajectory data.

Instead, we derive an exact hierarchy of moments directly from the
microscopic equation

    dx/dt = x + x^3 - x^5

For moments

    m_n = <x^n>

the exact hierarchy is

    dm_n/dt = n * (m_n + m_{n+2} - m_{n+4})

We then construct a natural Gaussian closure using only the first two
moments:

    mean = m1
    variance = m2 - m1^2

All higher moments are calculated from those two quantities.

The experiment compares:

    1. Full microscopic ensemble A
    2. Full microscopic ensemble B
    3. Gaussian moment closure

A and B are constructed to have the same initial mean and variance,
but different higher-order structure.

If A and B subsequently diverge, then the low-order variables are not
dynamically closed: the discarded higher moments contain information that
affects future evolution.

If the Gaussian closure becomes pathological while the full system remains
bounded, that would provide evidence for a representation-level pathology.

However, this experiment does NOT assume that such a pathology must occur.
A bounded closure would be an informative negative result.

Outputs
-------
results/006d_summary.txt
results/006d_moment_comparison.png
results/006d_hidden_state_difference.png
results/006d_distribution_evolution.png
results/006d_moment_hierarchy.png
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N = 200_000

DT = 0.002
T = 8.0

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RNG_SEED = 606


# ---------------------------------------------------------------------------
# Microscopic dynamics
# ---------------------------------------------------------------------------

def microscopic_rhs(x):
    """
    Microscopic dynamics.

        dx/dt = x + x^3 - x^5

    The -x^5 term dominates at large |x| and therefore prevents runaway
    growth of the underlying microscopic trajectories.
    """
    return x + x**3 - x**5


def rk4_step(x, dt):
    """
    One RK4 step for the microscopic ensemble.
    """
    k1 = microscopic_rhs(x)
    k2 = microscopic_rhs(x + 0.5 * dt * k1)
    k3 = microscopic_rhs(x + 0.5 * dt * k2)
    k4 = microscopic_rhs(x + dt * k3)

    return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# ---------------------------------------------------------------------------
# Moment calculations
# ---------------------------------------------------------------------------

def moments(x, maximum_order=10):
    """
    Calculate raw moments m_n = <x^n>.
    """
    return np.array(
        [np.mean(x**n) for n in range(1, maximum_order + 1)]
    )


def mean_variance(x):
    """
    Return first moment and variance.
    """
    m1 = np.mean(x)
    m2 = np.mean(x**2)

    variance = max(m2 - m1**2, 0.0)

    return m1, variance


# ---------------------------------------------------------------------------
# Gaussian raw moments
# ---------------------------------------------------------------------------

def gaussian_moments(mean, variance, maximum_order=10):
    """
    Raw moments of a Gaussian distribution.

    Uses the exact Gaussian moment formula

        E[(mu + sigma Z)^n]

    with Z ~ N(0,1).

    Only mean and variance are required.
    """

    result = np.zeros(maximum_order)

    for n in range(1, maximum_order + 1):

        total = 0.0

        for k in range(n + 1):

            # k = number of powers supplied by the zero-mean Gaussian part
            if k % 2 == 1:
                continue

            # Gaussian even moment:
            #
            # E[Z^k] = (k-1)!! for even k
            if k == 0:
                gaussian_moment = 1.0
            else:
                gaussian_moment = float(
                    np.prod(np.arange(k - 1, 0, -2))
                )

            total += (
                np.math.comb(n, k)
                * mean ** (n - k)
                * variance ** (k / 2.0)
                * gaussian_moment
            )

        result[n - 1] = total

    return result


# ---------------------------------------------------------------------------
# Gaussian closure dynamics
# ---------------------------------------------------------------------------

def gaussian_closure_rhs(mean, variance):
    """
    Natural Gaussian closure for the first two moments.

    Exact equations:

        dm1/dt = m1 + m3 - m5

        dm2/dt = 2 * (m2 + m4 - m6)

    The higher moments are supplied by the Gaussian distribution
    determined entirely by mean and variance.
    """

    gm = gaussian_moments(
        mean,
        variance,
        maximum_order=6,
    )

    m1 = gm[0]
    m2 = gm[1]
    m3 = gm[2]
    m4 = gm[3]
    m5 = gm[4]
    m6 = gm[5]

    dm1 = m1 + m3 - m5
    dm2 = 2.0 * (m2 + m4 - m6)

    dvariance = dm2 - 2.0 * mean * dm1

    return dm1, dvariance


def gaussian_closure_rk4_step(mean, variance, dt):
    """
    RK4 integration of the Gaussian moment closure.
    """

    def rhs(mu, var):
        return gaussian_closure_rhs(mu, max(var, 0.0))

    k1_mu, k1_var = rhs(mean, variance)

    k2_mu, k2_var = rhs(
        mean + 0.5 * dt * k1_mu,
        variance + 0.5 * dt * k1_var,
    )

    k3_mu, k3_var = rhs(
        mean + 0.5 * dt * k2_mu,
        variance + 0.5 * dt * k2_var,
    )

    k4_mu, k4_var = rhs(
        mean + dt * k3_mu,
        variance + dt * k3_var,
    )

    new_mean = mean + (dt / 6.0) * (
        k1_mu + 2*k2_mu + 2*k3_mu + k4_mu
    )

    new_variance = variance + (dt / 6.0) * (
        k1_var + 2*k2_var + 2*k3_var + k4_var
    )

    # Numerical protection against tiny negative variances.
    new_variance = max(new_variance, 0.0)

    return new_mean, new_variance


# ---------------------------------------------------------------------------
# Initial ensembles
# ---------------------------------------------------------------------------

def make_initial_ensembles(n, seed=606):
    """
    Construct two ensembles with identical mean and variance but different
    higher-order structure.

    Ensemble A:
        approximately Gaussian.

    Ensemble B:
        symmetric three-point-like mixture.

    Both are normalized to have exactly the same first two moments.
    """

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Ensemble A: Gaussian
    # ------------------------------------------------------------------

    A = rng.normal(
        loc=1.0,
        scale=0.25,
        size=n,
    )

    # ------------------------------------------------------------------
    # Ensemble B:
    #
    # A mixture with a small population farther from the mean.
    # The mixture is then affine-normalized so that mean and variance
    # exactly match A.
    # ------------------------------------------------------------------

    choices = rng.choice(
        [-1.0, 0.0, 1.0],
        size=n,
        p=[0.15, 0.70, 0.15],
    )

    B = 1.0 + 0.5 * choices

    # Match first two moments of A exactly.
    target_mean = np.mean(A)
    target_std = np.std(A)

    B_mean = np.mean(B)
    B_std = np.std(B)

    B = target_mean + (B - B_mean) * (target_std / B_std)

    return A, B


# ---------------------------------------------------------------------------
# Exact hierarchy diagnostic
# ---------------------------------------------------------------------------

def hierarchy_rhs_from_moments(m):
    """
    Evaluate the exact moment hierarchy

        dm_n/dt = n * (m_n + m_{n+2} - m_{n+4})

    for all moments for which the required higher moments are available.

    m contains moments m1 ... mN.

    Returns derivatives for m1 ... m(N-4).
    """

    N_moments = len(m)

    derivatives = np.zeros(N_moments)

    for n in range(1, N_moments - 3):

        mn = m[n - 1]
        mn2 = m[n + 1]
        mn4 = m[n + 3]

        derivatives[n - 1] = n * (
            mn + mn2 - mn4
        )

    return derivatives


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():

    print("=" * 72)
    print("Experiment 006d: Genuine Moment Hierarchy")
    print("=" * 72)

    print()
    print("Microscopic dynamics:")
    print("    dx/dt = x + x^3 - x^5")

    print()
    print("Exact moment hierarchy:")
    print("    dm_n/dt = n * (m_n + m_(n+2) - m_(n+4))")

    print()
    print("Effective closure:")
    print("    Gaussian closure using only mean and variance")

    # ------------------------------------------------------------------
    # Initial conditions
    # ------------------------------------------------------------------

    A, B = make_initial_ensembles(
        N,
        seed=RNG_SEED,
    )

    mean_A, var_A = mean_variance(A)
    mean_B, var_B = mean_variance(B)

    moments_A0 = moments(A, 10)
    moments_B0 = moments(B, 10)

    print()
    print("Initial conditions")
    print("-" * 72)

    print(
        f"Ensemble A mean      = {mean_A:.12f}"
    )
    print(
        f"Ensemble B mean      = {mean_B:.12f}"
    )

    print(
        f"Ensemble A variance  = {var_A:.12f}"
    )
    print(
        f"Ensemble B variance  = {var_B:.12f}"
    )

    print()
    print("Initial higher moments:")

    for n in range(3, 11):

        print(
            f"    m{n}: "
            f"A={moments_A0[n-1]: .8e}   "
            f"B={moments_B0[n-1]: .8e}"
        )

    # ------------------------------------------------------------------
    # Time arrays
    # ------------------------------------------------------------------

    steps = int(round(T / DT))
    times = np.arange(steps + 1) * DT

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    mean_A_history = np.zeros(steps + 1)
    mean_B_history = np.zeros(steps + 1)

    var_A_history = np.zeros(steps + 1)
    var_B_history = np.zeros(steps + 1)

    closure_mean_history = np.zeros(steps + 1)
    closure_var_history = np.zeros(steps + 1)

    moment_A_history = np.zeros((steps + 1, 10))
    moment_B_history = np.zeros((steps + 1, 10))

    max_abs_A = np.zeros(steps + 1)
    max_abs_B = np.zeros(steps + 1)

    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    closure_mean = mean_A
    closure_var = var_A

    for label, ensemble in [
        ("A", A),
        ("B", B),
    ]:
        pass

    mean_A_history[0] = mean_A
    mean_B_history[0] = mean_B

    var_A_history[0] = var_A
    var_B_history[0] = var_B

    closure_mean_history[0] = closure_mean
    closure_var_history[0] = closure_var

    moment_A_history[0] = moments_A0
    moment_B_history[0] = moments_B0

    max_abs_A[0] = np.max(np.abs(A))
    max_abs_B[0] = np.max(np.abs(B))

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    print()
    print("Evolving...")
    
    for step in range(1, steps + 1):

        A = rk4_step(A, DT)
        B = rk4_step(B, DT)

        closure_mean, closure_var = gaussian_closure_rk4_step(
            closure_mean,
            closure_var,
            DT,
        )

        # Ensemble moments
        mA = moments(A, 10)
        mB = moments(B, 10)

        mean_A, var_A = mean_variance(A)
        mean_B, var_B = mean_variance(B)

        mean_A_history[step] = mean_A
        mean_B_history[step] = mean_B

        var_A_history[step] = var_A
        var_B_history[step] = var_B

        closure_mean_history[step] = closure_mean
        closure_var_history[step] = closure_var

        moment_A_history[step] = mA
        moment_B_history[step] = mB

        max_abs_A[step] = np.max(np.abs(A))
        max_abs_B[step] = np.max(np.abs(B))

        if step % max(1, steps // 10) == 0:

            print(
                f"    t={times[step]:6.3f} "
                f"A_mean={mean_A: .6f} "
                f"B_mean={mean_B: .6f} "
                f"closure={closure_mean: .6f}"
            )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    mean_separation = np.abs(
        mean_A_history - mean_B_history
    )

    variance_separation = np.abs(
        var_A_history - var_B_history
    )

    closure_error_A = np.sqrt(
        (mean_A_history - closure_mean_history)**2
        +
        (var_A_history - closure_var_history)**2
    )

    closure_error_B = np.sqrt(
        (mean_B_history - closure_mean_history)**2
        +
        (var_B_history - closure_var_history)**2
    )

    combined_hidden_state_separation = np.sqrt(
        mean_separation**2
        +
        variance_separation**2
    )

    max_hidden_separation = np.max(
        combined_hidden_state_separation
    )

    final_hidden_separation = (
        combined_hidden_state_separation[-1]
    )

    max_closure_error_A = np.max(closure_error_A)
    max_closure_error_B = np.max(closure_error_B)

    final_closure_error_A = closure_error_A[-1]
    final_closure_error_B = closure_error_B[-1]

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------

    print()
    print("=" * 72)
    print("Results")
    print("=" * 72)

    print()
    print("Initial low-order agreement:")
    print(
        f"    |mean_A - mean_B|     = "
        f"{abs(mean_A_history[0] - mean_B_history[0]):.12e}"
    )
    print(
        f"    |variance_A - variance_B| = "
        f"{abs(var_A_history[0] - var_B_history[0]):.12e}"
    )

    print()
    print("Maximum microscopic amplitudes:")
    print(
        f"    Ensemble A = {np.max(max_abs_A):.8f}"
    )
    print(
        f"    Ensemble B = {np.max(max_abs_B):.8f}"
    )

    print()
    print("Hidden-state separation:")
    print(
        f"    maximum = {max_hidden_separation:.8e}"
    )
    print(
        f"    final   = {final_hidden_separation:.8e}"
    )

    print()
    print("Gaussian closure error:")
    print(
        f"    Ensemble A maximum = {max_closure_error_A:.8e}"
    )
    print(
        f"    Ensemble A final   = {final_closure_error_A:.8e}"
    )
    print(
        f"    Ensemble B maximum = {max_closure_error_B:.8e}"
    )
    print(
        f"    Ensemble B final   = {final_closure_error_B:.8e}"
    )

    # ------------------------------------------------------------------
    # Final moments
    # ------------------------------------------------------------------

    print()
    print("Final higher moments:")

    for n in range(3, 11):

        print(
            f"    m{n}: "
            f"A={moment_A_history[-1, n-1]: .8e}   "
            f"B={moment_B_history[-1, n-1]: .8e}"
        )

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------

    summary_path = OUTPUT_DIR / "006d_summary.txt"

    with summary_path.open("w", encoding="utf-8") as f:

        f.write("Experiment 006d: Genuine Moment Hierarchy\n")
        f.write("=" * 72 + "\n\n")

        f.write("Microscopic dynamics:\n")
        f.write("    dx/dt = x + x^3 - x^5\n\n")

        f.write("Exact moment hierarchy:\n")
        f.write("    dm_n/dt = n * (m_n + m_(n+2) - m_(n+4))\n\n")

        f.write("Effective closure:\n")
        f.write("    Gaussian closure using mean and variance\n\n")

        f.write("Initial conditions\n")
        f.write("-" * 72 + "\n")
        f.write(
            f"mean A      = {mean_A_history[0]:.12e}\n"
        )
        f.write(
            f"mean B      = {mean_B_history[0]:.12e}\n"
        )
        f.write(
            f"variance A  = {var_A_history[0]:.12e}\n"
        )
        f.write(
            f"variance B  = {var_B_history[0]:.12e}\n"
        )

        f.write("\nResults\n")
        f.write("-" * 72 + "\n")

        f.write(
            f"maximum hidden-state separation = "
            f"{max_hidden_separation:.12e}\n"
        )

        f.write(
            f"final hidden-state separation = "
            f"{final_hidden_separation:.12e}\n"
        )

        f.write(
            f"maximum closure error A = "
            f"{max_closure_error_A:.12e}\n"
        )

        f.write(
            f"final closure error A = "
            f"{final_closure_error_A:.12e}\n"
        )

        f.write(
            f"maximum closure error B = "
            f"{max_closure_error_B:.12e}\n"
        )

        f.write(
            f"final closure error B = "
            f"{final_closure_error_B:.12e}\n"
        )

        f.write("\nInterpretation\n")
        f.write("-" * 72 + "\n")

        f.write(
            "The two microscopic ensembles begin with identical mean and "
            "variance but different higher moments.\n"
        )

        f.write(
            "If their low-order moments subsequently separate, the first "
            "two moments are not dynamically closed.\n"
        )

        f.write(
            "The Gaussian closure deliberately discards the non-Gaussian "
            "higher-order information.\n"
        )

        f.write(
            "A discrepancy between the complete hierarchy and the closure "
            "therefore represents information lost by the effective "
            "description.\n"
        )

        f.write(
            "A singularity is NOT assumed or inserted in this experiment.\n"
        )

        f.write(
            "If the Gaussian closure remains regular, that is an important "
            "negative result: genuine hierarchy truncation alone did not "
            "produce a singularity in this model.\n"
        )

    # ------------------------------------------------------------------
    # Plot 1: first two moments
    # ------------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        times,
        mean_A_history,
        label="Full ensemble A: mean",
    )

    plt.plot(
        times,
        mean_B_history,
        label="Full ensemble B: mean",
    )

    plt.plot(
        times,
        closure_mean_history,
        "--",
        label="Gaussian closure: mean",
    )

    plt.xlabel("Time")
    plt.ylabel("Mean")
    plt.title("006d: Low-order dynamics")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "006d_moment_comparison.png",
        dpi=150,
    )

    plt.close()

    # ------------------------------------------------------------------
    # Plot 2: hidden-state separation
    # ------------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        times,
        combined_hidden_state_separation,
        label="A vs B low-order separation",
    )

    plt.xlabel("Time")
    plt.ylabel("Separation")
    plt.title(
        "006d: Divergence of initially identical low-order states"
    )
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "006d_hidden_state_difference.png",
        dpi=150,
    )

    plt.close()

    # ------------------------------------------------------------------
    # Plot 3: distributions at selected times
    # ------------------------------------------------------------------

    selected_times = [
        0.0,
        T * 0.25,
        T * 0.5,
        T,
    ]

    selected_indices = [
        int(round(t / DT))
        for t in selected_times
    ]

    # Reconstruct approximate distributions by rerunning the ensembles.
    # This keeps the main simulation storage compact.

    A_plot, B_plot = make_initial_ensembles(
        N,
        seed=RNG_SEED,
    )

    plt.figure(figsize=(10, 6))

    for target_index, target_time in zip(
        selected_indices,
        selected_times,
    ):

        while False:
            pass

        # Reset and advance to requested time.
        A_tmp, B_tmp = make_initial_ensembles(
            N,
            seed=RNG_SEED,
        )

        for _ in range(target_index):
            A_tmp = rk4_step(A_tmp, DT)
            B_tmp = rk4_step(B_tmp, DT)

        bins = np.linspace(
            -1.6,
            1.6,
            100,
        )

        hist_A, edges = np.histogram(
            A_tmp,
            bins=bins,
            density=True,
        )

        centres = 0.5 * (
            edges[:-1] + edges[1:]
        )

        plt.plot(
            centres,
            hist_A,
            label=f"A, t={target_time:.1f}",
        )

    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("006d: Evolution of microscopic ensemble A")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "006d_distribution_evolution.png",
        dpi=150,
    )

    plt.close()

    # ------------------------------------------------------------------
    # Plot 4: moment hierarchy
    # ------------------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for n in [1, 2, 3, 4, 5, 6]:

        plt.plot(
            times,
            np.abs(moment_A_history[:, n - 1]),
            label=f"|m{n}|",
        )

    plt.xlabel("Time")
    plt.ylabel("Absolute moment")
    plt.title("006d: Evolution of the exact moment hierarchy")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "006d_moment_hierarchy.png",
        dpi=150,
    )

    plt.close()

    print()
    print("Saved:")
    print(
        f"    {summary_path}"
    )
    print(
        "    results/006d_moment_comparison.png"
    )
    print(
        "    results/006d_hidden_state_difference.png"
    )
    print(
        "    results/006d_distribution_evolution.png"
    )
    print(
        "    results/006d_moment_hierarchy.png"
    )

    print()
    print("Experiment complete.")


if __name__ == "__main__":
    main()
