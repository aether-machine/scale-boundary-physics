# Scale Boundary Physics

## Working Hypothesis

**Status: exploratory research hypothesis.**

This project investigates whether the transition between physical scales can produce measurable effects that are absent from either description considered in isolation.

The immediate motivation is a general problem that appears whenever a microscopic physical system is represented by a lower-dimensional or coarse-grained description.

We ask:

> Does coarse-graining commute with physical evolution?

In mathematical form, consider a microscopic state \(X\), microscopic dynamics \(\Phi_{\mathrm{micro}}\), a coarse-graining map \(C\), and an effective dynamical model \(\Phi_{\mathrm{eff}}\).

The two possible routes are

$$
X_0
\overset{\Phi_{\mathrm{micro}}}{\longrightarrow}
X(t)
\overset{C}{\longrightarrow}
C[X(t)]
$$

and

$$
X_0
\overset{C}{\longrightarrow}
C[X_0]
\overset{\Phi_{\mathrm{eff}}}{\longrightarrow}
\Phi_{\mathrm{eff}}[C[X_0]].
$$

The central quantity is therefore

$$
\boxed{
\Delta(t)
=
C[\Phi_{\mathrm{micro}}(t,X_0)]
-
\Phi_{\mathrm{eff}}(t,C[X_0])
}
$$

We call this the **scale-boundary discrepancy**.

---

## 1. The Basic Question

A physical system can often be described at several levels.

For example:

```
microscopic particles
        ↓
kinetic / statistical description
        ↓
continuum fields
        ↓
macroscopic observables
```

Each description retains some information and discards other information.

The usual assumption is that the discarded information becomes irrelevant for the variables and scales of interest.

This is often an extremely good approximation.

The project asks what happens when that assumption begins to fail.

In particular:

> Can the boundary between two descriptions itself become dynamically significant?

---

## 2. What Is Established

Several aspects of the framework are standard physics and mathematics.

### 2.1 Coarse-graining is ubiquitous

Physical theories routinely replace microscopic degrees of freedom with macroscopic variables.

Examples include:

* molecular dynamics → kinetic theory;
* kinetic theory → hydrodynamics;
* microscopic spins → magnetization;
* atomic systems → elastic continua;
* microscopic degrees of freedom → thermodynamic state variables.

The resulting descriptions are generally not exact representations of every microscopic degree of freedom.

### 2.2 Effective theories are scale-dependent

An effective description is intended to reproduce relevant behaviour at a particular scale or under particular assumptions.

Consequently, a theory can be extremely successful within its domain without being complete at every scale.

### 2.3 Hydrodynamic equations can arise from microscopic descriptions

Under appropriate assumptions and limiting procedures, macroscopic fluid equations can be derived from more microscopic kinetic descriptions.

This provides an important precedent for the general architecture being investigated here.

### 2.4 Information is discarded by representation

A coarse-grained description contains fewer degrees of freedom than the microscopic system from which it was constructed.

The discarded variables may nevertheless influence future microscopic evolution.

This is the central mathematical reason to investigate whether

$$
C\circ\Phi_{\mathrm{micro}}
$$

and

$$
\Phi_{\mathrm{eff}}\circ C
$$

remain approximately equivalent.

---

## 3. The Working Hypothesis

The project's initial hypothesis is:

> **A coarse-grained physical description can remain highly accurate over a substantial range of scales while becoming increasingly inaccurate when the characteristic scale of the system approaches the scale at which the assumptions underlying the coarse-graining cease to hold.**

In that regime,

$$
|\Delta(t)|
$$

may increase substantially.

The important point is that this does **not** initially assume a physical singularity.

The first claim being investigated is weaker:

$$
\boxed{
\text{coarse-graining and evolution need not commute}
}
$$

The stronger questions come later.

---

## 4. Three Possible Interpretations

If a significant scale-boundary discrepancy is observed, several explanations remain possible.

### H1 — Genuine microscopic instability

The microscopic system itself develops an instability or singular behaviour.

In this case the effective description may be faithfully reflecting an underlying physical phenomenon.

### H2 — Inadequate effective closure

The microscopic dynamics remain regular, but the effective model is missing terms required to represent their influence.

The discrepancy is therefore a failure of the effective equation rather than of the underlying physics.

Schematically,

$$
\frac{\partial H}{\partial t}
=
F_{\mathrm{eff}}[H]
+
R[H,\xi],
$$

where \(\xi\) represents information discarded during coarse-graining.

### H3 — Scale-boundary dynamics

The transition between descriptions introduces behaviour that cannot be represented adequately by either description alone.

The effective dynamics may therefore require terms that become important only near a particular scale boundary.

Schematically,

$$
\frac{\partial H}{\partial t}
=
F_{\mathrm{eff}}[H]
+
R(H,L,\xi),
$$

where \(L\) represents characteristic scale.

This third possibility is the central speculative direction of the project.

---

## 5. Relation to Fluid Dynamics

The eventual physical target is the continuum description of fluids.

The Navier–Stokes equations are extraordinarily successful within their appropriate domain, but their mathematical behaviour under extreme extrapolation raises deep questions.

A finite-time singularity in a continuum description does not automatically imply that a physical fluid develops an infinite quantity.

Several possibilities must be distinguished:

1. the singularity corresponds to genuine microscopic physics;
2. the continuum approximation fails before the singularity;
3. the effective closure is incomplete;
4. additional degrees of freedom become dynamically relevant;
5. the mathematical singularity is an artefact of the idealized model.

This project does not assume which explanation is correct.

Instead, it asks whether a measurable breakdown of the correspondence

$$
C\circ\Phi_{\mathrm{micro}}
\approx
\Phi_{\mathrm{eff}}\circ C
$$

can be identified as a system approaches the limits of a continuum description.

---

## 6. A Useful Dimensionless Parameter

One natural measure of the validity of continuum descriptions is the Knudsen number,

$$
Kn=\frac{\lambda}{L},
$$

where

* \(\lambda\) is a characteristic microscopic mean free path;
* \(L\) is a characteristic macroscopic length scale.

Broadly:

$$
Kn\ll1
$$

corresponds to a regime where continuum descriptions can be effective.

As

$$
Kn\rightarrow1,
$$

microscopic structure becomes increasingly important.

This provides a potential bridge between the abstract scale-boundary discrepancy and physically meaningful scale transitions.

A future experiment should therefore investigate whether

$$
\Delta=\Delta(Kn).
$$

The important scientific question would be whether the discrepancy changes systematically as the scale separation changes.

---

## 7. First Computational Model

The first model in this repository is deliberately not a fluid.

It is a nonlinear one-dimensional periodic oscillator lattice.

The microscopic state is

$$
X=(q_1,p_1,\ldots,q_N,p_N),
$$

with nonlinear nearest-neighbour dynamics.

The microscopic system is divided into blocks, and each block is represented by coarse variables

$$
Q_j=\langle q\rangle_j,
\qquad
P_j=\langle p\rangle_j.
$$

The two routes are then evaluated independently.

### Route A

$$
X_0
\rightarrow
\Phi_{\mathrm{micro}}
\rightarrow
X(t)
\rightarrow
C[X(t)].
$$

### Route B

$$
X_0
\rightarrow
C[X_0]
\rightarrow
\Phi_{\mathrm{eff}}
\rightarrow
H(t).
$$

The experiment measures

$$
\Delta(t)=C[X(t)]-H(t).
$$

The purpose of this model is methodological rather than physical.

It asks whether the computational framework can detect a difference between evolution followed by coarse-graining and coarse-graining followed by independent evolution.

---

## 8. What Would Count as Interesting?

A non-zero discrepancy by itself would not be surprising.

Coarse-graining necessarily removes information, and an approximate effective model cannot generally reproduce every microscopic detail.

The more interesting result would be a systematic relationship between discrepancy and scale.

For example, we might find a regime such as

$$
\Delta\ll1
\qquad
L\gg\lambda
$$

followed by a transition toward

$$
\Delta=O(1)
\qquad
L\sim\lambda.
$$

That would suggest that the quality of the effective representation depends systematically on scale separation.

The stronger possibility would be that the transition has a recognizable mathematical structure and can be associated with a correction to the effective dynamics.

---

## 9. What Would Not Count as Evidence?

The project must avoid several common logical errors.

### A growing numerical error is not automatically a physical effect.

It could result from:

* numerical instability;
* poor timestep selection;
* an inadequate integrator;
* an inappropriate effective model;
* finite-size effects;
* boundary conditions;
* accumulated floating-point error.

### A continuum singularity is not automatically a physical singularity.

A mathematical divergence in an effective theory does not establish that a corresponding physical observable becomes infinite.

### A successful toy model does not establish a new physical theory.

The oscillator model is a test of methodology.

Any claim about fluids, quantum mechanics, or an underlying physical medium requires independent evidence.

---

## 10. The Deeper Hypothesis

The broader philosophical hypothesis motivating the project is that physical reality may be better represented not as a simple hierarchy of objects, but as a network of states and transformations between descriptions.

A conventional atomistic picture might be represented schematically as

```
objects
    ↓
smaller objects
    ↓
smaller objects
    ↓
...
```

The alternative being explored here is closer to

$$
\boxed{
\text{states}
+
\text{transformations}
+
\text{relationships between scales}
}
$$

In this view, the relationship between two levels of description may itself have physical significance.

This is currently a philosophical and modelling hypothesis, not an established physical principle.

---

## 11. Representation as a Physical Question

A useful analogy comes from digital systems.

A representation can preserve some properties while discarding others.

For example:

```
high-resolution state
        ↓
      encoding
        ↓
lower-resolution representation
        ↓
      decoding
```

The analogy suggests questions about:

* information preservation;
* resolution;
* sampling;
* aliasing;
* compression;
* state transitions;
* transformations between representations.

However, the project does **not** assume that the universe is literally a digital computer.

The digital analogy is being used as a conceptual and mathematical language for investigating transformations between descriptions.

---

## 12. Possible Extension Toward Underlying-Medium Models

A later stage of the project may investigate whether a proposed underlying medium can supply a physically motivated correction to an effective continuum equation.

Schematically,

$$
\frac{\partial u}{\partial t}
+
(u\cdot\nabla)u
=
-\frac{1}{\rho}\nabla p
+
\nu\nabla^2u
+
R_{\mathrm{medium}}.
$$

The key question would not be whether an arbitrary correction can be invented to remove a singularity.

The stronger question is:

> Does an independently motivated microscopic or sub-continuum theory naturally generate a correction that becomes important in the regime where the continuum description begins to fail?

Any candidate theory would therefore need to be tested against the same criteria as the simpler models.

---

## 13. Candidate Underlying Theories

One possible future direction is to investigate theories that posit additional sub-continuum dynamics.

Subquantum Kinetics is one candidate framework of interest because it attempts to describe physical phenomena through reaction-diffusion-like dynamics at a deeper level.

At present, this project does **not** assume that Subquantum Kinetics is correct.

The appropriate research question is narrower:

> Can an independently specified underlying-medium model produce effective corrections of the type required by a continuum description near a scale boundary?

This distinction is essential.

The project should test the framework rather than use the framework to pre-select the result.

---

## 14. Research Progression

The planned progression is:

```
001  Controlled toy system
     ↓
002  Scale dependence and block size
     ↓
003  Nonlinear effective closure
     ↓
004  Kinetic description
     ↓
005  Hydrodynamic / continuum limit
     ↓
006  Navier–Stokes boundary behaviour
     ↓
007  Candidate regularizing corrections
     ↓
008  Candidate underlying-medium theories
```

Each stage should preserve the same basic methodology:

1. define the microscopic state;
2. define the representation map;
3. define the effective dynamics;
4. evolve both routes independently;
5. measure the discrepancy;
6. identify its dependence on scale;
7. test alternative explanations.

---

## 15. Falsifiability

The hypothesis should be considered unsuccessful if the apparent scale-boundary effects can be completely explained by ordinary numerical error, poor modelling choices, or known corrections without any remaining systematic phenomenon.

It should also be weakened if

$$
\Delta
$$

does not exhibit reproducible dependence on scale separation.

Conversely, the hypothesis becomes more interesting if:

1. the discrepancy is robust under numerical refinement;
2. it survives changes in the effective-model parametrization;
3. it scales systematically with a physical dimensionless parameter;
4. the transition occurs consistently near a meaningful scale boundary;
5. a physically motivated correction can predict the discrepancy;
6. the correction generalizes across different systems.

The strongest possible result would be a new effective term whose form is derived independently and which predicts when the conventional description will fail.

---

## 16. Current Status

This repository is exploratory.

The project currently establishes a computational language for studying relationships between microscopic and effective descriptions.

It does **not** establish:

* a new theory of quantum mechanics;
* a solution to the Navier–Stokes Millennium Prize Problem;
* a physical subquantum medium;
* a digital ontology of nature;
* or a replacement for atomistic physics.

Those are possible questions for future investigation, not conclusions of the present work.

The immediate goal is much simpler:

$$
\boxed{
\text{Measure what happens when representation and evolution are performed in different orders.}
}
$$

If that effect can be characterized mathematically and physically, we can then ask whether it provides a useful lens for understanding the limits of continuum descriptions.
