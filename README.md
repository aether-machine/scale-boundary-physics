# Scale Boundary Physics

## Investigating dynamic relationships between physical representations

This repository explores a simple question:

> **Can a physical system remain well behaved at one level of description while its effective mathematical representation becomes inaccurate or pathological as the system approaches a boundary of scale?**

The immediate motivation is the Navier–Stokes singularity problem, but the project does **not** assume that Navier–Stokes is incorrect, nor that any particular underlying-medium theory is correct.

The initial investigation is deliberately more general.

---

## 1. The central idea

Physical systems can be represented at different levels:

$$
\text{microscopic dynamics}
\rightarrow
\text{kinetic description}
\rightarrow
\text{continuum description}.
$$

For example:

$$
X
\rightarrow
f(x,v,t)
\rightarrow
(\rho,\mathbf u,T).
$$

Each transition is a form of coarse-graining: information that is not required at the higher level is discarded.

Normally we expect the resulting description to remain dynamically faithful.

The central question is whether this remains true as the characteristic scale of the system approaches the scale at which information was discarded.

---

## 2. Evolution versus representation

Let

$$
\Phi_{\rm micro}(t)
$$

represent microscopic evolution and let

$$
C
$$

represent coarse-graining.

There are two ways to obtain a coarse representation at time \(t\).

### Route A — evolve first

$$
X_0
\overset{\Phi_{\rm micro}}{\longrightarrow}
X(t)
\overset{C}{\longrightarrow}
C[X(t)].
$$

### Route B — coarse-grain first

$$
X_0
\overset{C}{\longrightarrow}
C[X_0]
\overset{\Phi_{\rm eff}}{\longrightarrow}
\Phi_{\rm eff}(t)C[X_0].
$$

If the effective description is dynamically faithful, these should approximately agree:

$$
C[\Phi_{\rm micro}(t)X_0]
\approx
\Phi_{\rm eff}(t)C[X_0].
$$

We define the difference as

$$
\boxed{
\Delta_t =
C[\Phi_{\rm micro}(t)X_0]
-
\Phi_{\rm eff}(t)C[X_0]
}
$$

and call it the **representation-transition error**.

The first goal of this project is to measure this quantity.

---

## 3. The first hypothesis

The working hypothesis is:

$$
|\Delta_t|\ll1
$$

under ordinary conditions, but that the discrepancy may increase systematically as the characteristic physical scale approaches the scale removed by coarse-graining.

This is only a hypothesis.

A result showing that the discrepancy does **not** behave this way is equally valuable.

---

## 4. What we are not assuming

This project does not initially assume:

* that Navier–Stokes is mathematically flawed;
* that the Navier–Stokes singularity is physically real;
* that continuum physics must fail;
* that an underlying medium exists;
* that Subquantum Kinetics is correct;
* that quantum mechanics is caused by scale transitions;
* that physical reality is fundamentally digital.

Those are possible later hypotheses.

The first task is much narrower:

> **Can we experimentally/numerically characterize the relationship between microscopic evolution and its coarse-grained representations?**

---

## 5. Research progression

The project is intended to proceed through increasingly realistic models.

### Experiment 001 — Controlled toy system

Construct a microscopic system for which the dynamics are known and bounded.

Compare:

$$
C[\Phi_{\rm micro}(X_0)]
$$

with

$$
\Phi_{\rm eff}[C(X_0)].
$$

The objective is to establish the methodology.

### Experiment 002 — Nonlinear microscopic dynamics

Introduce nonlinear interactions and investigate whether the representation-transition error develops qualitatively different behaviour.

### Experiment 003 — Kinetic description

Compare microscopic particle dynamics with a kinetic description such as the Boltzmann equation.

### Experiment 004 — Hydrodynamic limit

Compare kinetic dynamics with the corresponding hydrodynamic equations.

Relevant dimensionless parameters will include the Knudsen number:

$$
Kn=\frac{\lambda}{L}.
$$

### Experiment 005 — Navier–Stokes boundary

Investigate whether the approach toward pathological behaviour in the continuum equations corresponds to a measurable increase in the discrepancy between microscopic/kinetic and continuum descriptions.

### Experiment 006 — Candidate correction

Attempt to derive the effective correction required by the lower-level description:

$$
\text{Navier–Stokes}
+
R_{\rm boundary}.
$$

The correction must be derived from the underlying model rather than introduced merely to remove a singularity.

### Experiment 007 — Candidate underlying-medium theories

Only after the preceding stages should specific theories such as Subquantum Kinetics be evaluated.

The question then becomes:

> **Does the proposed theory independently predict a correction resembling the one required by the observed multiscale transition?**

---

## 6. The deeper hypothesis

The broader philosophical hypothesis motivating the project is that physical theories may not simply form a hierarchy of increasingly small objects.

Instead, physical reality may involve:

$$
\boxed{
\text{states}
+
\text{transformations}
+
\text{relationships between scales}
}
$$

The transitions between descriptions may therefore be more than mathematical bookkeeping.

They may have measurable consequences.

This would amount to a revised form of atomism:

> Physical reality may consist of underlying degrees of freedom together with dynamically meaningful relationships between different levels of description.

This is speculative and is not assumed by the experiments.

---

## 7. Connection to Navier–Stokes

Navier–Stokes is an especially useful test case because it is an extraordinarily successful continuum description of fluid behaviour.

The question is therefore not:

> "Is Navier–Stokes bad mathematics?"

Instead:

> **What happens when an exceptionally successful continuum representation is extrapolated toward a regime where its underlying assumptions may cease to hold?**

If a singularity occurs, several explanations remain possible:

1. The singularity is a genuine feature of the continuum equations.
2. The continuum equations remain valid and the singularity has a physical interpretation we do not yet understand.
3. The continuum approximation breaks down before the mathematical singularity.
4. Microscopic or kinetic degrees of freedom generate a correction.
5. A deeper level of physical dynamics becomes relevant.
6. The apparent problem is ultimately mathematical rather than physical.

The purpose of this project is to distinguish these possibilities rather than presuppose one.

---

## 8. Scientific standard

A proposed explanation should satisfy increasingly demanding tests.

A useful candidate should ideally:

1. arise independently of the phenomenon it is being used to explain;
2. reproduce known physics in the appropriate limit;
3. predict when its corrections become important;
4. make quantitative predictions;
5. produce results that can be compared with simulation or experiment;
6. allow the hypothesis to be falsified.

A correction that merely prevents an infinity after being fitted to the desired result is not sufficient evidence for new physics.

---

## 9. First objective

The first computational objective is therefore:

$$
\boxed{
\text{Determine whether evolution and coarse-graining commute,
and map where they stop doing so.}
}
$$

Only after that map exists should we ask what physical mechanism produces the departure.

---

## Status

**Stage:** exploratory computational research

**Version:** 0.1

**Interpretation:** speculative framework; not an established physical theory.
