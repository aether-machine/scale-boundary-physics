# Experimental Results

## 1. Experimental philosophy

These experiments investigate a general question motivated by the mathematical behaviour of the three-dimensional incompressible Navier–Stokes equations:

> **What happens when a dynamical system is represented at a lower level of description than the level at which its underlying dynamics are closed?**

The original motivation was the possibility that a finite-time singularity in a continuum field description might indicate a limitation of the representation rather than a corresponding physical singularity.

The experiments therefore do **not** attempt to reproduce or explain the Navier–Stokes singularity directly. Instead, they investigate simpler systems in which the relationship between microscopic dynamics, coarse-grained variables, and effective dynamics can be controlled explicitly.

The central diagnostic is the discrepancy between two operations:

$$\(\boxed{\Delta_t = C\left[\Phi_{\mathrm{micro}}(t)X_0\right] - \Phi_{\mathrm{eff}}(t)C[X_0]}\)$$

where:

- $$\(X_0\)$$ is a microscopic initial state,
- $$\(\Phi_{\mathrm{micro}}\)$$ is the underlying microscopic evolution,
- $$\(C\)$$ is a coarse-graining or projection operation,
- $$\(\Phi_{\mathrm{eff}}\)$$ is the proposed effective evolution,
- $$\(\Delta_t\)$$ measures the failure of the effective description to reproduce the coarse-grained microscopic dynamics.

The project has progressively examined several possible sources of this discrepancy:

1. coarse spatial resolution;
2. hidden microscopic structure;
3. kinetic versus hydrodynamic descriptions;
4. relaxation times and characteristic scales;
5. explicit nonlocality;
6. projection-induced non-closure;
7. truncated moment hierarchies;
8. higher-order stabilising information;
9. memory and history dependence;
10. the possibility of predicting closure uncertainty from coarse variables and their histories.

The purpose is not to establish that coarse-graining necessarily produces singularities. Rather, the aim is to determine under what conditions an effective description ceases to be dynamically closed, and what additional structure is required to repair it.

---

## 2. 001–002: Baseline effective dynamics

### 2.1 Experiment 001 — Micro versus effective dynamics

The first experiment established a minimal microscopic/effective comparison.

A periodic nonlinear lattice system was evolved at the microscopic level and compared with an independently evolved coarse effective system.

The maximum discrepancy between the two descriptions was approximately:

$$\[
\Delta_{\max}\approx1.20.
\]$$

The discrepancy grew substantially from an initially negligible value.

### Interpretation

This established that two descriptions which begin from corresponding initial states need not remain dynamically equivalent.

However, the experiment had an important methodological limitation: the effective model reused the microscopic coefficients on the coarse lattice. It therefore demonstrated divergence between the chosen models, but did not establish that the discrepancy was specifically caused by a scale boundary.

This motivated the more controlled experiments in 002.

---

### 2.2 Experiment 002 — Hidden structure and resolution

Experiment 002 introduced microscopic structure that could be filtered by coarse-graining.

The initial sweep showed that discrepancy depended strongly on both block size and the scale of the microscopic perturbation.

In particular, larger hidden perturbations and larger coarse-graining blocks generally produced larger differences.

The strongest case in the controlled 002b sweep occurred for a hidden amplitude of \(0.40\) and wavelength \(32\), where the maximum discrepancy increased from approximately:

$$\[
1.07 \quad (B=2)
\]$$

to

$$\[
2.29 \quad (B=16).
\]$$

### Important correction

The initial experiments revealed that coarse-graining scale and perturbation wavelength could easily become confounded.

Experiment 002c therefore examined the ratio between the perturbation wavelength and block size more explicitly.

The result was not a simple monotonic "closer to the boundary means larger error" relationship.

Some long-wavelength perturbations remained well resolved and nevertheless produced large discrepancies, while sufficiently fine perturbations could be strongly filtered out.

### Interpretation

The important lesson from 001–002 was therefore not that coarse-graining automatically produces increasing error.

Rather:

> **The effect of coarse-graining depends on the relationship between the retained variables, the unresolved structure, and the dynamics acting on that unresolved structure.**

---

## 3. 002d — Hidden-state dependence

Experiment 002d provided a more direct test of dynamical closure.

Two microscopic states were constructed so that they produced essentially identical initial coarse states while differing internally within each coarse block.

Initially:

$$\[
\|C(X_A)-C(X_B)\|\approx 4.6\times10^{-17}.
\]$$

Despite this near-identical coarse representation, the subsequent coarse trajectories diverged.

For the representative block-size-2, amplitude-0.40 case:

$$\[
\Delta_{\max}\approx0.369
\]$$

and the final separation was approximately:

$$\[
\Delta_{\mathrm{final}}\approx0.367.
\]$$

### Interpretation

This provides a clean demonstration of a basic closure problem:

> Two microscopic states can have the same retained coarse variables but different future coarse evolution.

In other words, the coarse variables alone do not necessarily contain enough information to define an exact autonomous dynamical system.

This observation became central to the later experiments.

---

## 4. 003–004: Microscopic → kinetic → hydrodynamic

### 4.1 Experiment 003 — Particle to kinetic description

The next stage moved from a lattice model toward a particle/kinetic description.

A microscopic particle ensemble was represented by a phase-space distribution obtained through spatial and velocity binning.

The resulting kinetic representation preserved expected macroscopic quantities, including approximately uniform mean density and stable velocity statistics.

The basic phase-space representation was therefore numerically well behaved.

### 4.2 Experiment 003b — Hidden microscopic states

Experiment 003b constructed two microscopic ensembles with identical initial kinetic histograms but different microscopic arrangements.

The initial kinetic $$\(L^2\)$$ difference was exactly zero, while the microscopic states differed substantially.

During evolution, however, their kinetic representations separated.

The maximum kinetic $$\(L^2\)$$ separation was approximately:

$$\[
0.444.
\]$$

The maximum difference in magnetisation was approximately:

$$\[
0.102.
\]$$

### Interpretation

This reproduced the earlier closure phenomenon at a more physically motivated level:

> **A kinetic distribution can be identical while the underlying microscopic state is not, and those hidden differences can subsequently influence the kinetic evolution.**

The result does not imply that kinetic theory is invalid. It demonstrates that a finite-resolution kinetic representation can itself fail to be dynamically closed if the retained variables do not capture all dynamically relevant information.

---

## 5. 004 — Kinetic to hydrodynamic closure

Experiment 004 introduced a BGK-like kinetic model with a hydrodynamic projection.

Two kinetic states were constructed to have approximately identical hydrodynamic moments while differing in their higher velocity-space structure.

Initially, the hydrodynamic differences were at numerical roundoff level, while the kinetic $$\(L^2\)$$ difference was approximately:

$$\[
1.83.
\]$$

During evolution, the hidden kinetic difference propagated into the hydrodynamic variables.

Maximum differences were approximately:

| Quantity | Maximum difference |
|---|---:|
| Density | 0.220 |
| Velocity | 0.103 |
| Temperature | 0.470 |
| Combined hydrodynamic measure | 0.523 |

The final combined difference was approximately:

$$\[
0.345.
\]$$

### 5.1 Relaxation dependence

Experiment 004b varied the kinetic relaxation time \(\tau\).

The maximum hydrodynamic discrepancy increased from approximately:

$$\[
0.499 \quad (\tau=0.05)
\]$$

to:

$$\[
0.563 \quad (\tau=1.0).
\]$$

The final discrepancy increased from approximately:

$$\[
0.245
\]$$

to:

$$\[
0.446.
\]$$

Experiment 004c extended the relaxation sweep toward smaller \(\tau\).

The final discrepancy decreased progressively as relaxation became faster.

### Interpretation

This is consistent with the expectation that rapid kinetic relaxation suppresses the persistence of unresolved velocity-space information.

However, this is a BGK analogue rather than a derivation of the Navier–Stokes equations. It should therefore be interpreted as a controlled model of closure, not as direct evidence about real fluids.

---

## 6. 005: Scale and relaxation

Experiment 005 investigated whether closure discrepancy was controlled primarily by characteristic spatial scale.

The initial sweep varied the wavelength of the hydrodynamic perturbation while keeping other parameters fixed.

The result was not the originally expected simple increase in discrepancy with decreasing length scale.

Instead, discrepancy depended strongly on relaxation time and numerical resolution.

### 6.1 Controlled length/relaxation scaling

Experiment 005b linked relaxation time to characteristic length:

$$\[
\tau=\epsilon\frac{L}{v_{\mathrm{ref}}}.
\]$$

At fixed $$\(\epsilon\)$$, the final hydrodynamic discrepancy generally decreased as $$\(L\)$$ decreased.

For example, at $$\(\epsilon=0.05\)$$:

$$\[
\Delta_{\mathrm{final}}
=
0.382,\;0.309,\;0.248,\;0.200,\;0.179,\;0.166,\;0.146
\]$$

as $$\(k\)$$ increased from 1 to 24.

This initially appeared suggestive of a scale effect.

### 6.2 Grid convergence

Experiment 005c showed that the observed behaviour was reasonably stable under grid refinement for representative cases.

For example, at \(k=8,\epsilon=0.05\), the final discrepancy was approximately:

$$\[
0.0991,\;0.0998,\;0.1002
\]$$

for $$\(N_X=128,256,512\)$$.

This reduced concern that the observed effect was purely a grid-resolution artefact.

However, the parameter sweep still coupled $$\(L\)$$ and $$\(\tau\)$$.

---

## 7. 005d — Separating characteristic length from relaxation time

Experiment 005d varied $$\(L\)$$ and $$\(\tau\)$$ independently.

This was an important correction to the earlier interpretation.

At fixed $$\(L\)$$, increasing $$\(\tau\)$$ produced a strong and generally monotonic increase in final closure discrepancy.

At fixed $$\(\tau\)$$, changing $$\(L\)$$ also mattered, particularly at larger characteristic lengths.

For example, at $$\(\tau=0.05\)$$:

| \(L\) | Final discrepancy |
|---:|---:|
| 6.283 | 0.198 |
| 1.571 | 0.123 |
| 0.785 | 0.120 |
| 0.393 | 0.119 |

The result also showed that equal values of a dimensionless ratio such as \(\tau/L\) did not uniquely determine the discrepancy.

For example, cases with approximately equal \(\tau/L\) could differ by factors of roughly \(2.5\)–\(3\).

### Interpretation

This is an important negative result.

It argues against reducing closure error to a single dimensionless scale parameter in this model.

More generally:

> **The existence of a characteristic scale ratio does not by itself imply a universal closure law.**

The detailed dynamics, relaxation mechanism, and representation all matter.

---

## 8. 005f–005g: Explicit microscopic length and nonlocality

### 8.1 Experiment 005f — Explicit microscopic length

The next model introduced an explicit spatial smoothing length $$\(a\)$$ into the kinetic relaxation operator.

The local relaxation model used the local distribution, while the modified model used a spatially smoothed distribution.

The intention was to introduce a controlled analogue of a microscopic interaction length and investigate whether closure behaviour changed as the hydrodynamic scale $$\(L\)$$ approached it.

The experiments did not reveal a sharp transition at $$\(L/a\approx1\)$$.

Instead, closure differences varied smoothly with the parameters.

### Interpretation

This is another useful negative result.

An explicit microscopic length does not automatically produce a sharp "continuum breaks here" transition.

The way microscopic information enters the effective dynamics matters.

---

### 8.2 Experiment 005g — Local versus nonlocal closure

Experiment 005g compared:

$$\(\frac{Df}{Dt} = \frac{f_{\mathrm{eq}}[f]-f}{\tau}\)$$

with a nonlocal variant:

$$\(\frac{Df}{Dt} = \frac{f_{\mathrm{eq}}[\bar{f}_a]-f}{\tau}\)$$

The resulting operator difference varied substantially with scale separation.

The maximum instantaneous operator difference ranged from approximately:

$$\[
6.6\times10^{-3}
\]$$

to:

$$\[
3.27.
\]$$

The final hydrodynamic discrepancy ranged from approximately:

$$\[
6.3\times10^{-5}
\]$$

to:

$$\[
0.102.
\]$$

### Interpretation

This provides a controlled demonstration that introducing a finite spatial interaction scale can produce a **scale-dependent correction to an effective local theory**.

Importantly, however:

- there was no singularity;
- there was no universal threshold;
- equal values of $$\(L/a\)$$ did not completely collapse the results;
- the nonlocal operator was a deliberately chosen model.

The experiment therefore supports the possibility of scale-dependent corrections, but does not establish that such corrections regularise Navier–Stokes singularities.

---

## 9. 006b–006c: Projection and pathological closures

### 9.1 Experiment 006b — Projection-induced non-closure

A bounded harmonic oscillator,

$$\[
\dot{x}=y,\qquad
\dot{y}=-x,
\]$$

was projected onto $$\(x\)$$ alone.

The underlying dynamics are exactly regular.

However, $$\(x\)$$ does not obey a closed first-order autonomous equation of the form

$$\[
\dot{x}=F(x).
\]$$

The exact reduced description is instead:

$$\[
\ddot{x}+x=0.
\]$$

Polynomial approximations to $$\(F(x)\)$$ failed to remove the fundamental problem: the same value of $$\(x\)$$ can occur with different values of $$\(\dot{x}\)$$.

### Interpretation

This experiment demonstrated a crucial distinction:

> **Projection can destroy dynamical closure without producing a singularity.**

Therefore, non-closure alone is not sufficient to explain pathological effective dynamics.

---

### 9.2 Experiment 006c — Closure-induced runaway

A second bounded system was constructed with stabilising nonlinear dynamics:

$$\[
\dot{x}=y,
\]$$

$$\[
\epsilon\dot{y}=x+x^3-x^5-y.
\]$$

The $$\(-x^5\)$$ term provides strong stabilisation at large amplitude.

Closures trained only over $$\(|x|\le1\)$$ were then extrapolated from an initial condition outside the training region.

Low-order closures could become catastrophically unstable.

For example, with \(\epsilon=0.05\), a degree-1 closure reached a magnitude of approximately:

$$\[
|x|\sim10^6.
\]$$

The underlying system remained bounded.

Higher-order closures performed substantially better; the degree-5 closure remained bounded with prediction RMS error approximately $$\(0.065\)$$.

### Interpretation

This experiment provides a controlled example of an important mechanism:

> **A regular underlying system can be represented by an effective closure that develops qualitatively pathological behaviour when dynamically important stabilising structure has been omitted or incorrectly extrapolated.**

However, this does **not** demonstrate that coarse-graining inherently creates singularities.

The pathology depended on an inadequate closure and extrapolation outside its training regime.

---

## 10. 006d–006f: Moment hierarchy and discarded stabilisation

### 10.1 Experiment 006d — Genuine moment hierarchy

The microscopic system

$$\[
\dot{x}=x+x^3-x^5
\]$$

produces an exact moment hierarchy:

$$\(\dot{m}_n = n\left(m_n + m_{n+2} - m_{n+4}\right)\)$$

Two ensembles were constructed with identical low-order moments but different higher moments.

Their retained mean and variance agreed initially, while their subsequent low-order dynamics differed.

The maximum hidden-state separation in the retained variables was approximately:

\[
0.0137.
\]

The maximum Gaussian-closure error was approximately:

\[
0.0296.
\]

### Interpretation

Higher moments can influence the evolution of lower moments.

But the closure error in this example remained bounded and eventually decayed.

Again:

> **Discarded information does not automatically produce pathological behaviour.**

---

### 10.2 Experiment 006e — Stabilising contribution of discarded moments

The same hierarchy was deliberately truncated by setting higher moments to zero.

This removed the stabilising contribution associated with terms such as \(m_5\).

The full microscopic system remained bounded, with:

\[
\max |x|\approx1.54.
\]

The truncated hierarchy instead developed runaway growth.

The \(m_1\) error eventually reached approximately:

\[
2.25\times10^{15}.
\]

The runaway threshold \(|m_1|>10\) was reached at approximately:

\[
t=4.59.
\]

### Interpretation

This gives a particularly clear example of how omitted higher-order information can contain **stabilising dynamics**.

The effective model was not simply less accurate numerically. It had qualitatively different long-term behaviour.

The result is nevertheless a deliberately crude truncation and produces runaway rather than a finite-time singularity.

That distinction is important.

---

### 10.3 Experiment 006f — Principled closure can recover stabilisation

Experiment 006f tested a more principled Gaussian closure.

The maximum closure error was approximately:

\[
5.63\times10^{-3},
\]

while the maximum error in the reconstructed \(m_5\) was approximately:

\[
1.76\times10^{-2}.
\]

The microscopic system remained bounded:

\[
\max |x|\approx1.61.
\]

The effective closure did not fail catastrophically.

### Interpretation

This experiment provides an important counterexample to an overly strong version of the hypothesis.

The mere fact that information has been discarded does not determine the behaviour of the effective model.

A sufficiently accurate closure can reconstruct the dynamically important information.

Thus the more precise principle emerging from these experiments is:

\[
\boxed{
\text{discarded information}
\neq
\text{automatic pathology}
}
\]

Instead:

\[
\boxed{
\text{discarded information}
+
\text{insufficient closure}
\rightarrow
\text{possible qualitative failure}
}
\]

---

## 11. 006g — Intrinsic non-closure

Experiment 006g asked whether the retained variables themselves can fail to determine the exact future.

Two ensembles were constructed with essentially identical retained moments:

\[
\Delta m_1\approx2.2\times10^{-16},
\]

\[
\Delta m_2=0.
\]

Despite this, their exact derivatives differed by as much as:

\[
0.057.
\]

The maximum separation in the retained variables during evolution was approximately:

\[
0.0055.
\]

Both underlying microscopic systems remained bounded.

### Interpretation

This provides a particularly clean demonstration of **intrinsic non-closure**.

If two microscopic states have the same retained state but different exact derivatives, then no exact autonomous law of the form

\[
\dot H=F(H)
\]

can describe both systems.

Additional information is required.

That information might take the form of:

- higher moments;
- hidden variables;
- correlations;
- spatial nonlocality;
- memory/history;
- stochastic variables;
- or some other extended state representation.

---

## 12. 006h–006i: Memory and temporal information

### 12.1 Experiment 006h — Hand-designed memory closure

A simple memory variable was added to the reduced model.

The memory closure did not improve the prediction.

The maximum error was approximately:

\[
0.0164,
\]

compared with approximately:

\[
0.0154
\]

for the instantaneous closure.

The memory model was therefore approximately \(6.4\%\) worse at its maximum error.

### Interpretation

This is a useful negative result.

It demonstrates that adding "memory" in the abstract is not sufficient.

An arbitrary memory ansatz can make a closure worse.

The experiment also had methodological limitations, including the hand-chosen memory equation and differences in numerical integration.

It should therefore not be interpreted as evidence against memory as a closure mechanism.

---

### 12.2 Experiment 006i — Data-driven memory closure

Experiment 006i instead learned a finite-history closure from independent trajectories.

The instantaneous model used:

\[
\dot H=F(H_t).
\]

The memory model used:

$$\(\dot{H} = A_0H_t + A_1H_{t-\tau} + \cdots + A_kH_{t-k\tau}\)$$

Out-of-sample test error was:

| Model | Test RMS error |
|---|---:|
| Instantaneous | 0.06276 |
| Best memory model | 0.05597 |

The absolute improvement was:

\[
0.00679,
\]

or approximately:

\[
10.8\%.
\]

The best model used 16 history steps.

### Interpretation

This is stronger evidence that temporal information can improve closure than 006h, because the memory relationship was learned from data and evaluated out of sample.

It demonstrates:

> **Recent coarse history can contain predictive information about coarse dynamics that is absent from the instantaneous coarse state.**

This does not establish that memory is fundamental.

Nor does it establish a particular physical memory kernel.

The next question is whether the predictive improvement corresponds to a measurable reduction in conditional dynamical uncertainty.

---

## 13. 006j — Hidden-state reconstruction

Experiment 006j tested a different question:

> Can coarse history identify which hidden microscopic state generated a trajectory?

A classifier was trained using ordered coarse histories and compared with shuffled histories.

The classifier remained essentially at chance.

The best ordered-history accuracy was approximately:

\[
0.504.
\]

The ordered-history advantage over the shuffled baseline was approximately:

\[
-0.00013.
\]

### Interpretation

There was no evidence that the tested coarse history could reconstruct the hidden-state identity.

This is not contradictory to Experiment 006i.

Prediction and hidden-state reconstruction are different tasks.

A history can contain enough information to improve prediction of a future derivative without containing enough information to uniquely identify the microscopic state that generated the trajectory.

This distinction is important for the broader hypothesis.

It suggests that a useful reduced description need not recover the microscopic state itself.

It may only need to retain the information relevant to predicting the future evolution of the retained variables.

---

## 14. 006k — Conditional closure uncertainty

Experiment 006k asked whether the dynamical uncertainty of the retained variables could be reduced by conditioning on their recent history.

The retained variables were:

$$\[
H=(m_1,m_2),
\]$$

where

$$\[
m_1=E[x],
\qquad
m_2=E[x^2].
\]$$

The microscopic dynamics were:

$$\[
\dot{x}=x+x^3-x^5.
\]$$

The quantity of interest was the conditional variance of the microscopic coarse derivative:

$$\(\mathcal{U}_0 = \mathrm{Var}\left(\dot{H}\mid H\right)\)$$

and, for increasingly long histories,

$$\(\mathcal{U}_k = \mathrm{Var}\left(\dot{H}\mid H_t,H_{t-1},\ldots,H_{t-k}\right)\)$$

The motivation was straightforward.

If the instantaneous coarse state is not dynamically closed, then different microscopic states can produce the same \(H_t\) while having different coarse derivatives.

However, some of the information distinguishing those microscopic states may be encoded in the recent trajectory of \(H\).

If so, conditioning on history should reduce the remaining uncertainty in \(\dot H\).

### Results

The estimated conditional uncertainty decreased systematically with history length:

| History length | Conditional uncertainty | Bootstrap SD | Fractional reduction |
|---:|---:|---:|---:|
| 0 | 0.13773 | 0.00272 | 0.0% |
| 1 | 0.12319 | 0.00341 | 10.6% |
| 2 | 0.11609 | 0.00371 | 15.7% |
| 4 | 0.09811 | 0.00265 | 28.8% |
| 8 | 0.07900 | 0.00137 | 42.6% |
| 16 | 0.05683 | 0.00161 | 58.7% |

The reduction is considerably larger than the bootstrap uncertainty at each stage.

At history length 16, the estimated uncertainty was approximately:

\[
\mathcal U_{16}=0.0568,
\]

compared with:

\[
\mathcal U_0=0.1377.
\]

Thus the estimated unexplained variance was reduced by approximately:

\[
58.7\%.
\]

### Interpretation

This provides evidence that recent coarse history contains dynamical information that is absent from the instantaneous coarse state.

This is consistent with the intrinsic non-closure observed in 006g and with the improved out-of-sample prediction obtained from the learned memory model in 006i.

The result can be interpreted as follows.

The instantaneous coarse state

\[
H_t
\]

does not completely determine the coarse derivative.

But the extended state

\[
(H_t,H_{t-1},\ldots,H_{t-k})
\]

contains additional information about that derivative.

In this toy system, temporal information therefore makes the effective representation **more dynamically closed**.

This does not establish that memory is fundamental to physical dynamics. It establishes a narrower and testable result:

> **When the instantaneous coarse variables are insufficient for dynamical closure, recent coarse history can contain information that reduces the remaining uncertainty in their evolution.**

The monotonic decrease across the tested history lengths is particularly suggestive:

\[
\mathcal U_0
>
\mathcal U_1
>
\mathcal U_2
>
\mathcal U_4
>
\mathcal U_8
>
\mathcal U_{16}.
\]

However, the current experiment does not yet establish the characteristic memory timescale. The best history length tested was 16, so it is not known whether the uncertainty has reached a plateau or would continue to decrease with longer histories.

### Relationship to Experiment 006i

Experiment 006i found that a data-driven finite-history model reduced out-of-sample derivative prediction error by approximately 10.8%.

Experiment 006k asks a somewhat more fundamental statistical question: whether conditioning on history reduces the conditional uncertainty itself.

The two results are therefore complementary.

Experiment 006i showed:

> History improves prediction.

Experiment 006k shows:

> History reduces the unexplained conditional variation in the coarse derivative.

Together they provide stronger evidence that temporal information is relevant to the closure problem than either result alone.

### Important caveat

The conditional uncertainty was estimated using nearest-neighbour methods.

As the history length increases, the effective dimension of the conditioning space also increases. This can make non-parametric conditional-variance estimates statistically difficult.

The bootstrap uncertainties reported here provide an indication of estimator variability, but they do not eliminate all finite-sample or high-dimensional estimation effects.

The result should therefore be treated as evidence for a history-dependent closure structure rather than as a precise measurement of a physical memory kernel.

### Next question

The next experiment should determine whether the apparent reduction continues, saturates, or reverses at longer history lengths.

A particularly useful test would compare:

\[
k=0,1,2,4,8,16,32,64,\ldots
\]

while controlling sample size and estimator bias.

If the uncertainty approaches a stable plateau, that would provide evidence for a finite effective memory timescale in this model.

If it continues decreasing over increasingly long histories, the appropriate reduced description may require a longer-memory representation.

Either outcome would help determine what additional structure is required to make the coarse dynamics approximately closed.

---
# 15. Effective Closure, Hidden State, and Memory

The 006 series asks a more specific question than the earlier scale-transition experiments:

> What happens when a dynamical system is represented using variables that do not contain enough information to determine their own future evolution?

The central distinction is between **information loss** and **dynamical closure**.

Coarse-graining necessarily removes information. That alone is not a problem. An effective description can remain perfectly valid if the retained variables contain enough information to predict their own evolution.

The potential problem arises when the retained variables are insufficient to determine the future state.

Schematically, let $$\(X\)$$ denote the full microscopic state and $$\(H=C[X]\)$$ a reduced or coarse-grained state. The full system evolves according to some map $$\(\Phi_t^{\mathrm{micro}}\)$$, while the effective model uses $$\(\Phi_t^{\mathrm{eff}}\)$$.

The central diagnostic is whether coarse-graining and evolution approximately commute:

$$\(\boxed{\Delta_t = C\left[\Phi_{\mathrm{micro}}(t)X_0\right] - \Phi_{\mathrm{eff}}(t)C[X_0]}\)$$

If $$\(\Delta_t\)$$ becomes significant, the effective representation is failing to reproduce the dynamics of the underlying system.

The experiments below progressively distinguish several possible causes of this failure:

1. instantaneous coarse variables may not be dynamically closed;
2. low-order closures can omit stabilising information;
3. higher moments can sometimes restore part of that information;
4. temporal history can also restore predictive information;
5. history need not identify the hidden microscopic state itself;
6. conditional closure uncertainty can quantify the remaining ambiguity;
7. matched-complexity controls can test whether the apparent benefit of history is simply a consequence of giving the model more parameters.

The purpose of the series is not to demonstrate that memory is the unique mechanism of effective closure.

It is to determine whether **finite history contains predictive information that is not efficiently represented by the tested instantaneous variables**.

---

## 15.1 Intrinsic Non-Closure

### 006g — Intrinsic Non-Closure

The first question was whether two microscopic ensembles could have the same retained state while having different future dynamics.

The microscopic system was

$$\(\dot{x}=x+x^3-x^5\)$$.

The retained variables were the first two moments,

$$\(H=(m_1,m_2)\)$$,

where $$\(m_n=\mathbb{E}[x^n]\)$$.

Two different microscopic ensembles were constructed so that they initially had the same \(m_1\) and \(m_2\), while differing in their higher-order structure.

The initial retained states agreed to numerical precision:

- initial \(m_1\) difference: approximately \(2.2\times10^{-16}\);
- initial \(m_2\) difference: \(0\);
- initial retained-state separation: \(0\).

Despite this, the subsequent retained dynamics diverged.

The maximum retained-state separation was approximately

\(5.53\times10^{-3}\),

while the maximum difference between the corresponding microscopic derivative information was approximately

\(5.70\times10^{-2}\).

The maximum difference in the hidden microscopic state was approximately

\(2.96\times10^{-2}\).

Both microscopic systems remained bounded.

### Interpretation

This provides a direct toy-model demonstration of **intrinsic non-closure**.

The same instantaneous retained state can correspond to different microscopic states and therefore different future derivatives.

Consequently, there is no exact autonomous first-order evolution law of the form

\(\dot{H}=F(H)\)

for \(H=(m_1,m_2)\) alone in this system.

This is stronger than merely observing prediction error in a fitted model. It demonstrates a structural ambiguity in the reduced variables themselves.

The important point is that non-closure did **not** produce a singularity.

The underlying dynamics remained regular.

Therefore:

> Non-closure is a possible source of effective-model failure, but non-closure by itself does not imply pathological or singular dynamics.

---

## 15.2 Memory as Additional Predictive Information

### 006h — Hand-Chosen Memory Closure

The next experiment tested whether adding a simple memory variable could improve an instantaneous closure.

A memory-augmented effective model was constructed using the retained state together with a hand-chosen relaxation variable.

The maximum instantaneous closure error was approximately

\(1.54\times10^{-2}\).

The maximum error of the memory-augmented closure was approximately

\(1.64\times10^{-2}\).

The maximum error reduction was therefore negative, approximately

\(-9.9\times10^{-4}\),

corresponding to a roughly \(6.4\%\) deterioration at the maximum error point.

### Interpretation

This experiment is important primarily as a negative result.

A memory term does not automatically improve an effective description.

The memory equation used here was hand-chosen rather than derived from the microscopic dynamics. The result therefore does not test whether memory is fundamentally useful; it tests one particular memory ansatz.

The negative result prevents an overly simple conclusion:

> Adding a memory variable is not, by itself, sufficient to restore closure.

A more general data-driven approach was therefore used next.

---

### 006i — Data-Driven Memory Closure

The next experiment compared an instantaneous model,

\(\dot{H}=F(H_t)\),

with models given access to a finite history,

\(\dot{H}=A_0H_t+A_1H_{t-\tau}+\cdots+A_kH_{t-k\tau}\).

The models were trained on independent trajectories and evaluated out of sample.

The instantaneous model achieved a test RMS error of approximately

\(6.2764\times10^{-2}\).

The best finite-history model achieved approximately

\(5.5974\times10^{-2}\).

This corresponds to an improvement of approximately \(10.8\%\).

The best result occurred at the largest history length tested, \(k=16\).

### Interpretation

Unlike the hand-chosen memory experiment, the data-driven experiment found that temporal history improved prediction.

The result indicates that:

> The recent history of the coarse variables contains predictive information about their future evolution that is not fully captured by the instantaneous retained state.

This does not establish that the underlying physical dynamics are fundamentally non-Markovian.

A finite history can act as a proxy for many kinds of unresolved information, including:

- hidden microscopic variables;
- higher moments;
- unresolved correlations;
- phase information;
- latent state variables;
- slowly relaxing degrees of freedom.

The result therefore establishes a predictive fact rather than an ontological interpretation.

---

## 15.3 Does History Recover the Hidden State?

### 006j — Hidden-State Reconstruction

A natural follow-up question was whether the predictive value of history came from reconstructing which hidden microscopic state the system was actually in.

Two matched microscopic ensembles were constructed with the same initial retained state but different hidden structure.

A classifier was then given increasingly long histories of the retained variables and asked to distinguish the two hidden-state classes.

The classifier remained close to chance.

Representative results were:

| History length | Ordered accuracy | Shuffled accuracy |
|---:|---:|---:|
| 0 | 0.5038 | 0.5039 |
| 1 | 0.5020 | 0.5017 |
| 2 | 0.5011 | 0.5011 |
| 4 | 0.5005 | 0.5275 |
| 8 | 0.4983 | 0.4983 |
| 16 | 0.4985 | 0.4992 |

The best ordered-history accuracy was only approximately \(0.504\).

### Interpretation

There was no evidence that the tested history reconstructed the hidden-state identity.

This is not inconsistent with 006i.

The two experiments ask different questions.

006i asks:

> Does history improve prediction of the future derivative?

006j asks:

> Does history reveal which of two particular hidden microscopic ensembles generated the observation?

A representation can improve prediction without uniquely identifying the underlying microscopic state.

This distinction is important.

Effective closure does not necessarily require reconstructing the complete hidden state. It may only require recovering enough information to predict the future evolution of the retained variables.

---

## 15.4 Conditional Closure Uncertainty

### 006k — Conditional Closure Uncertainty

A more direct measure of closure was introduced by examining the variation in the coarse derivative that remains after conditioning on the information available to the effective model.

For instantaneous closure, define

\(\mathcal{U}_0=\operatorname{Var}\left(\dot{H}\mid H\right)\).

For a history of length \(k\), define

\(\mathcal{U}_k=\operatorname{Var}\left(\dot{H}\mid H_t,H_{t-1},\ldots,H_{t-k}\right)\).

The measured uncertainty decreased systematically as history was added:

| History length | Conditional uncertainty | Bootstrap standard deviation | Fractional reduction |
|---:|---:|---:|---:|
| 0 | 0.13773 | 0.00272 | 0.0% |
| 1 | 0.12319 | 0.00341 | 10.6% |
| 2 | 0.11609 | 0.00371 | 15.7% |
| 4 | 0.09811 | 0.00265 | 28.8% |
| 8 | 0.07900 | 0.00137 | 42.6% |
| 16 | 0.05683 | 0.00161 | 58.7% |

At history length 16, the estimated unexplained variance was reduced by approximately \(58.7\%\) relative to instantaneous conditioning.

### Interpretation

This provides a more direct measure of the closure problem than prediction error alone.

The instantaneous retained state leaves substantial ambiguity in the future derivative.

Adding temporal information progressively reduces that ambiguity.

The result is consistent with the hypothesis that the coarse state is not dynamically sufficient on its own and that some of the missing predictive information is recoverable from recent history.

The result should nevertheless be interpreted cautiously.

Conditional-variance estimation becomes more difficult as the dimension of the conditioning space increases. Nearest-neighbour estimates can suffer from finite-sample effects and high-dimensional sparsity.

The bootstrap measures variability of the estimator under resampling; it does not remove all possible estimator bias.

The experiment therefore demonstrates a robust trend rather than identifying a unique physical memory kernel.

---

# 15.5 What Kind of Information Restores Closure?

The previous experiments established that history can contain predictive information.

The next question was more demanding:

> Is temporal history actually providing something that instantaneous higher moments cannot provide?

This matters because the apparent role of memory could simply reflect the fact that the retained variables \(m_1,m_2\) were too limited.

If adding sufficiently many instantaneous moments produced the same predictive improvement, there would be less reason to describe the effect specifically in terms of memory.

Two experiments addressed this.

---

## 15.5.1 Higher-Moment Closure

### 006l — Minimal Sufficient Closure

The first comparison expanded the instantaneous state using higher moments and compared this with adding history.

The models included:

- \(m_1,m_2\);
- \(m_1,m_2\) plus 4 history points;
- \(m_1,m_2\) plus 16 history points;
- \(m_1,m_2,m_3\);
- \(m_1,\ldots,m_4\);
- \(m_1,\ldots,m_5\);
- \(m_1,m_2,m_3\) plus 4 history points.

The results were:

| Representation | Features | Test RMS | \(R^2\) |
|---|---:|---:|---:|
| \(m_1,m_2\) | 2 | 0.055245 | 0.901950 |
| \(m_1,m_2\) + history 4 | 10 | 0.051682 | 0.909350 |
| \(m_1,m_2\) + history 16 | 34 | 0.040528 | 0.923867 |
| \(m_1,m_2,m_3\) | 3 | 0.054673 | 0.903968 |
| \(m_1,\ldots,m_4\) | 4 | 0.054654 | 0.904035 |
| \(m_1,\ldots,m_5\) | 5 | 0.054654 | 0.904036 |
| \(m_1,m_2,m_3\) + history 4 | 15 | 0.051682 | 0.909350 |

The instantaneous higher-moment representations produced only small improvements over the \(m_1,m_2\) baseline.

By contrast, adding history produced progressively larger improvements.

For example, the 16-history representation reduced RMS error by approximately \(26.6\%\) relative to the instantaneous \(m_1,m_2\) baseline.

### Interpretation

Within the tested model class, adding higher moments through \(m_5\) produced relatively little additional predictive information, whereas adding temporal history produced substantially more.

However, this comparison was not yet complexity-matched.

A history model with many lagged variables contains more regression parameters than a model containing only a few instantaneous moments.

The result therefore motivates a stricter control.

---

## 15.5.2 Matched-Complexity Closure

### 006m — Matched-Complexity Closure

The next experiment compared moment-based and history-based closures using the same regression framework while approximately matching model complexity.

The comparisons included:

- \(m_1,m_2\) versus \(m_1,m_2\) plus one history point;
- \(m_1,\ldots,m_6\) versus \(m_1,m_2\) plus two history points;
- \(m_1,\ldots,m_8\) versus \(m_1,m_2\) plus three history points;
- and longer matched representations.

The initial results were:

| Representation | Features | Parameters | RMS |
|---|---:|---:|---:|
| \(m_1,m_2\) | 2 | 4 | 0.055084 |
| \(m_1,\ldots,m_6\) | 6 | 12 | 0.054491 |
| \(m_1,\ldots,m_8\) | 8 | 16 | 0.054491 |
| \(m_1,m_2\) + history 1 | 4 | 8 | 0.053818 |
| \(m_1,m_2\) + history 2 | 6 | 12 | 0.053096 |
| \(m_1,m_2\) + history 4 | 10 | 20 | 0.051529 |
| \(m_1,m_2\) + history 8 | 18 | 36 | 0.048054 |

The result suggested that history continued to improve prediction even after adding several instantaneous moments.

However, because this first version used a single train/test split, a multi-split equal-budget experiment was performed.

---

# 15.6 Equal-Budget Memory vs Moment Closure

### 006n — Equal-Budget Memory vs Moment Closure

This experiment was designed as the strongest control in the 006 series.

Moment-based and history-based representations were given equal numbers of regression parameters.

The experiment used:

- 8 independent train/test splits;
- 1000 independent trajectories per split;
- 700 training trajectories;
- 300 test trajectories;
- 512 particles per ensemble;
- \(dt=0.01\);
- total simulation time \(T=8\);
- the same ridge-regression framework for every representation.

The matched representations were:

| Parameter budget | Moment representation | History representation |
|---:|---|---|
| 8 | \(m_1,\ldots,m_4\) | \(m_1,m_2\) + history 1 |
| 12 | \(m_1,\ldots,m_6\) | \(m_1,m_2\) + history 2 |
| 16 | \(m_1,\ldots,m_8\) | \(m_1,m_2\) + history 3 |
| 20 | \(m_1,\ldots,m_{10}\) | \(m_1,m_2\) + history 4 |
| 28 | \(m_1,\ldots,m_{14}\) | \(m_1,m_2\) + history 6 |
| 36 | \(m_1,\ldots,m_{18}\) | \(m_1,m_2\) + history 8 |

The aggregate results were:

| Representation | Budget | Mean RMS | Std. RMS | Mean \(R^2\) |
|---|---:|---:|---:|---:|
| \(m_1,\ldots,m_4\) | 8 | 0.054610 | 0.000033 | 0.903976 |
| \(m_1,m_2\) + history 1 | 8 | 0.053934 | 0.000032 | 0.905296 |
| \(m_1,\ldots,m_6\) | 12 | 0.054609 | 0.000033 | 0.903977 |
| \(m_1,m_2\) + history 2 | 12 | 0.053208 | 0.000032 | 0.906624 |
| \(m_1,\ldots,m_8\) | 16 | 0.054609 | 0.000033 | 0.903977 |
| \(m_1,m_2\) + history 3 | 16 | 0.052440 | 0.000031 | 0.907950 |
| \(m_1,\ldots,m_{10}\) | 20 | 0.054609 | 0.000033 | 0.903977 |
| \(m_1,m_2\) + history 4 | 20 | 0.051636 | — | 0.909268 |
| \(m_1,\ldots,m_{14}\) | 28 | 0.054609 | 0.000033 | 0.903977 |
| \(m_1,m_2\) + history 6 | 28 | 0.049939 | 0.000029 | 0.911863 |
| \(m_1,\ldots,m_{18}\) | 36 | 0.054609 | 0.000033 | 0.903977 |
| \(m_1,m_2\) + history 8 | 36 | 0.048149 | 0.000028 | 0.914389 |

The matched-pair comparisons showed the following reductions in RMS error for the history representation:

| Budget | Mean RMS difference | Fractional history advantage |
|---:|---:|---:|
| 8 | 0.000676 | 1.24% |
| 12 | 0.001401 | 2.57% |
| 16 | 0.002169 | 3.97% |
| 20 | 0.002973 | 5.45% |
| 28 | 0.004670 | 8.55% |
| 36 | 0.006460 | 11.83% |

The history representation had lower RMS error on all eight train/test splits at every tested parameter budget.

### Interpretation

This is the strongest result of the 006 series.

At matched regression capacity, finite temporal history consistently improved prediction relative to the tested instantaneous moment representations.

The effect was not confined to a single train/test split.

The most directly interpretable comparison is the 12-parameter case:

- instantaneous \(m_1,\ldots,m_6\): mean RMS \(\approx0.05461\);
- \(m_1,m_2\) plus two history points: mean RMS \(\approx0.05321\).

The relative reduction in RMS error was approximately \(2.6\%\).

At the largest tested budget, 36 parameters, the corresponding reduction was approximately \(11.8\%\).

The moment-only models were also notable for their lack of improvement.

From \(m_1,\ldots,m_4\) through \(m_1,\ldots,m_{18}\), the mean RMS remained approximately \(0.05461\).

By contrast, the history models improved monotonically over the tested range.

This suggests that the predictive information contained in recent history is not efficiently represented by simply adding the tested instantaneous moments.

---

# 15.7 What the 006 Series Establishes

Taken together, experiments 006g–006n provide a coherent progression.

### 1. Instantaneous coarse variables can be intrinsically non-closed

006g showed that two microscopic ensembles can have identical retained variables but different future derivatives.

Therefore an exact autonomous equation

\(\dot{H}=F(H)\)

need not exist for a chosen coarse representation.

### 2. Memory is not automatically beneficial

006h produced a negative result for a hand-designed memory ansatz.

This prevents the conclusion that any memory term will restore closure.

### 3. Data-driven history can improve prediction

006i showed that finite history reduced out-of-sample prediction error relative to instantaneous closure.

### 4. Predictive memory does not necessarily identify hidden microscopic state

006j found no evidence that the tested history reconstructed the identity of the hidden ensemble.

Thus predictive sufficiency and hidden-state identification are distinct problems.

### 5. History reduces conditional closure uncertainty

006k showed a systematic reduction in conditional derivative uncertainty, reaching approximately \(58.7\%\) reduction at the longest tested history.

### 6. Adding low-order instantaneous moments produced only modest improvement

006l found little additional predictive value from extending the instantaneous representation through \(m_5\).

### 7. The history result survives a complexity control

006m and especially 006n showed that the predictive benefit of history persists when moment-based and history-based models are given matched regression capacity.

---

# 15.8 The Current Interpretation

The experiments support the following working interpretation:

> **A coarse-grained dynamical description can fail to be Markovian in its instantaneous variables because unresolved information continues to influence its future evolution. Some of that information can be represented either by additional state variables or, approximately, by finite temporal history.**

This is a statement about the structure of the effective description.

It is not a claim that memory is necessarily fundamental.

History can encode information about hidden variables without revealing those variables explicitly.

A useful conceptual picture is therefore:

\(\text{microscopic state}\rightarrow\text{coarse state}\)

followed, when closure is incomplete, by

\(\text{coarse state}+\text{history}\rightarrow\text{improved effective prediction}\).

An alternative is to enlarge the instantaneous state:

\(\text{coarse state}+\text{additional variables}\rightarrow\text{improved closure}\).

The two approaches are not fundamentally different in principle. A sufficiently rich set of additional state variables can convert a history-dependent process into a higher-dimensional Markovian one.

The experiments therefore do not establish that time-history is ontologically fundamental.

They establish that **temporal context is a useful carrier of unresolved predictive information in this model**.

---

# 15.9 Relationship to the Scale-Boundary Hypothesis

The 006 experiments are deliberately more abstract than the earlier fluid and kinetic experiments.

They do not simulate the Navier–Stokes singularity.

They instead isolate a mechanism that could, in principle, become relevant when moving between physical descriptions.

The proposed chain is:

\(\text{microscopic dynamics}\rightarrow\text{coarse-graining}\rightarrow\text{loss of information}\rightarrow\text{loss of closure}\rightarrow\text{effective correction required}\).

That correction need not be a single “memory term”.

Depending on the underlying system, the missing structure could take the form of:

- additional moments;
- hidden variables;
- correlations;
- nonlocal interactions;
- stochastic terms;
- memory kernels;
- or combinations of these.

The experiments therefore support a more general hypothesis than “memory fixes singularities”.

The broader hypothesis is:

> **When an effective description ceases to contain enough information to remain dynamically closed, additional structure may become necessary.**

Memory is one possible representation of that additional structure.

---

# 15.10 Important Negative Result: No Singularity Was Produced

None of the 006 experiments demonstrates a finite-time singularity in the underlying dynamics.

This distinction is essential.

Several experiments produced runaway behaviour in deliberately truncated or badly extrapolated effective models.

For example, 006c showed that a low-order closure can become catastrophically unstable even when the underlying system remains bounded.

006e similarly showed that discarding the stabilising higher-order contribution can produce runaway moment dynamics.

But these are not demonstrations of a Navier–Stokes-type finite-time singularity.

They demonstrate something more limited:

> A regular underlying system can admit an effective model whose extrapolation becomes qualitatively pathological because dynamically important structure has been omitted.

That is directly relevant to the broader hypothesis, but it should not be overstated.

---

# 15.11 What These Experiments Do Not Establish

The 006 series does **not** establish any of the following:

1. that the Navier–Stokes equations actually develop a physically realised singularity;
2. that molecular discreteness necessarily regularises such a singularity;
3. that kinetic theory provides the missing physics in the Navier–Stokes problem;
4. that memory is the unique or fundamental mechanism of effective closure;
5. that higher moments are generally inferior to history;
6. that the universe is fundamentally hierarchical in the proposed philosophical sense;
7. that quantum mechanics represents a scale-boundary phenomenon;
8. that any particular proposed underlying theory, including Subquantum Kinetics, supplies the required missing structure.

The experiments are evidence for a narrower proposition:

> **Reduced variables can fail to form a dynamically closed state representation, and temporal or additional state information can sometimes restore predictive closure.**

---

# 15.12 Limitations

Several limitations remain.

### Toy dynamics

The principal 006 system,

\(\dot{x}=x+x^3-x^5\),

is deliberately simple.

Its purpose is to isolate closure effects, not to reproduce fluid turbulence.

### Finite moment hierarchy

The moment hierarchy provides a useful analytical control, but only finitely many moments were tested in the computational comparisons.

For example,

\(\dot{m}_n=n(m_n+m_{n+2}-m_{n+4})\).

Consequently, even apparently large instantaneous moment representations do not constitute an exact closure of the hierarchy.

### Model-class dependence

The conclusions concern the tested ridge-regression models and representations.

Another model class could exploit the instantaneous variables more effectively.

### History length

The best history length in 006i and 006k occurred at the largest tested value.

This means that a characteristic memory timescale or saturation point has not yet been identified.

### Conditional-variance estimation

The conditional uncertainty calculations are subject to finite-sample and high-dimensional estimation effects.

The bootstrap quantifies resampling variability but does not eliminate all estimator bias.

### Hidden-state reconstruction

Failure to reconstruct the particular A/B hidden-state identity does not prove that hidden information cannot be inferred from history.

It only shows that the tested classifier and representation did not reliably distinguish those particular hidden ensembles.

### No physical singularity

Most importantly, none of these experiments connects the toy closure mechanism quantitatively to the proposed Navier–Stokes singularity.

That connection remains an open research question.

---

# 15.13 Current Working Model

The current conceptual model can therefore be summarised as:

\(\boxed{\text{discarded information}\neq\text{pathology}}\)

but potentially:

\(\boxed{\text{discarded information}+\text{insufficient closure}\rightarrow\text{effective-model failure}}\)

and, when the missing information is correlated with recent evolution,

\(\boxed{\text{instantaneous state}+\text{history}\rightarrow\text{improved predictive closure}}\).

This gives the project a more precise formulation than the original intuition.

The question is no longer simply:

> “Does coarse-graining lose information?”

It obviously does.

The more interesting question is:

> **When does the information removed by coarse-graining remain dynamically irrelevant, and when does it become necessary for a closed effective description?**

That is the question that the subsequent work should connect back to physical scale transitions.

---

# 15.14 Status of the 006 Series

The 006 series has now established a controlled toy-model framework for studying dynamical closure.

The main empirical progression is:

\(\text{same coarse state}\)

\(\rightarrow\)

\(\text{different hidden futures}\)

\(\rightarrow\)

\(\text{instantaneous non-closure}\)

\(\rightarrow\)

\(\text{history improves prediction}\)

\(\rightarrow\)

\(\text{conditional uncertainty decreases}\)

\(\rightarrow\)

\(\text{history remains useful under matched complexity}\).

The strongest current result is therefore not that memory is the answer.

It is that **the choice of state representation matters dynamically**.

A representation that is sufficient for describing a system at one level may not remain sufficient for predicting its evolution after information has been discarded.

The next stage of the project should therefore focus on connecting this closure framework back to explicit physical scale transitions, while preserving the distinction between:

- mathematical singularities;
- failures of effective models;
- failures of closure;
- and genuine physical breakdown of a continuum description.


# 16. Cross-experiment synthesis

The experiments now suggest a more precise picture than the original hypothesis.

The initial intuition was:

\[
\text{information discarded}
\rightarrow
\text{singularity}.
\]

The experiments do not support that statement.

A better formulation is:

\[
\boxed{
\text{microscopic state}
\rightarrow
\text{coarse representation}
\rightarrow
\text{closure problem}
}
\]

followed by a possible need for additional structure:

\[
\boxed{
\begin{cases}
\text{higher moments}\\
\text{hidden variables}\\
\text{memory}\\
\text{nonlocality}\\
\text{correlations}\\
\text{stochastic structure}
\end{cases}
}
\]

The important observation is that the boundary between descriptions can become dynamically significant when the retained variables cease to provide a closed representation.

A concise statement of the current hypothesis is therefore:

> **The important issue is not simply that information is discarded. The issue is whether the retained representation remains dynamically closed, and what additional structure is required when it does not.**

---

## 17. What the experiments currently establish

Within the toy and simplified kinetic models studied here, the experiments demonstrate that:

1. coarse representations can diverge from microscopic dynamics;
2. hidden microscopic structure can influence later coarse evolution;
3. identical coarse states need not have identical futures;
4. kinetic-to-hydrodynamic closure depends on relaxation and scale;
5. explicit finite-scale/nonlocal corrections can alter effective dynamics;
6. projection can destroy autonomous closure without producing a singularity;
7. inadequate closures can produce qualitatively pathological behaviour even when the underlying system remains bounded;
8. omitted higher-order information can contain stabilising dynamics;
9. sufficiently good closures can recover that stabilisation;
10. arbitrary memory does not necessarily improve a closure;
11. data-driven finite history can improve out-of-sample prediction;
12. predictive history need not uniquely reconstruct hidden microscopic state.

Taken together, these results support the general study of **dynamical closure across levels of description**.

---

# 18. What the experiments do not establish

The experiments do **not** establish that:

- the claimed Navier–Stokes singularity is physically realised;
- molecular discreteness necessarily regularises a Navier–Stokes singularity;
- continuum field theory is fundamentally wrong;
- the Navier–Stokes singularity is caused by coarse-graining;
- quantum mechanics is fundamentally a scale-boundary phenomenon;
- nature is fundamentally digital or sampled;
- Subquantum Kinetics provides the underlying correction;
- a particular microscopic theory explains the Navier–Stokes problem;
- coarse-graining generically produces finite-time singularities.

The connection to Navier–Stokes remains a motivation and a future test case, not an established conclusion.

---

# 19. Current limitations

Several limitations remain.

### 19.1 Toy models

Most experiments use deliberately simplified dynamical systems.

They are useful because microscopic states and coarse-graining operations can be controlled, but they cannot substitute for a physically grounded kinetic-to-hydrodynamic derivation.

### 19.2 Closure dependence

Some experiments use deliberately chosen closures.

A pathological closure can demonstrate that effective descriptions *can* fail, but it does not show that the failure occurs naturally in a particular physical theory.

### 19.3 Numerical effects

Resolution, timestep, interpolation, finite sampling, and estimator choice can all influence measured closure error.

Grid-convergence tests are therefore required whenever scale-dependent conclusions are drawn.

### 18.4 No singularity reproduction

The project has not yet produced a genuine finite-time singularity emerging naturally from a regular microscopic model through coarse-graining.

The runaway behaviour observed in 006c and 006e is not equivalent to finite-time blow-up.

### 18.5 No microscopic length-scale theorem

Introducing an explicit microscopic length produced scale-dependent corrections, but no universal threshold or sharp transition.

A true physical claim about continuum breakdown will require a better-founded model.

---

# 20. Questions remaining

The next experiments should address several increasingly demanding questions:

1. Does coarse history measurably reduce conditional dynamical uncertainty?
2. Can the required memory structure be inferred rather than imposed?
3. When does memory outperform adding higher moments?
4. Can an effective model remain stable over long times while reproducing microscopic dynamics?
5. Can these ideas be reproduced in a more physically grounded kinetic theory?
6. Can an explicit microscopic length be connected to a controlled hydrodynamic limit?
7. Under what conditions does a continuum effective description develop qualitatively incorrect behaviour?
8. Can a correction term be **derived** from the underlying dynamics rather than fitted?
9. Does any such mechanism survive in a system whose continuum limit is closely related to Navier–Stokes?

Only after those questions are addressed should candidate underlying-medium theories be introduced.

---

# 21. Working conceptual model

The current working picture is:

\[
\boxed{
\text{micro}
\rightarrow
\text{kinetic}
\rightarrow
\text{hydrodynamic}
\rightarrow
\text{macroscopic}
}
\]

but with the possibility that the relationships between these levels become dynamically important.

The deeper conceptual hypothesis can therefore be expressed as:

\[
\boxed{
\text{physical description}
=
\text{states}
+
\text{transformations}
+
\text{relationships between scales}
}
\]

This is currently a philosophical and methodological hypothesis, not a claim about fundamental ontology.

The scientific programme is to determine whether measurable closure failures provide evidence for treating those inter-level relationships as part of the effective dynamics themselves.

---

# 22. Current status

The project remains exploratory.

The strongest result so far is not evidence for a particular explanation of the Navier–Stokes singularity.

It is evidence for a more general phenomenon:

> **A lower-dimensional or coarse-grained representation can cease to be dynamically closed even while the underlying dynamics remain regular.**

The most promising next step is therefore not to assume what the missing physics is, but to determine experimentally what kind of additional structure is required:

\[
\boxed{
\text{higher moments?}
\quad
\text{hidden variables?}
\quad
\text{memory?}
\quad
\text{nonlocality?}
\quad
\text{something else?}
}
\]

The central question remains open:

> **When an effective physical description approaches the limits of its validity, does the apparent pathology reveal a failure of the underlying physics—or a failure of the description to retain the variables and relationships that have become dynamically relevant?**
