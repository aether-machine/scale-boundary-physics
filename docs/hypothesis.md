# Scale Boundary Physics

## Working Hypothesis

**Status: exploratory research hypothesis.**

This project investigates whether the transition between physical scales can
produce measurable effects that are absent from either description considered
in isolation.

The immediate motivation is a general problem that appears whenever a
microscopic physical system is represented by a lower-dimensional or
coarse-grained description.

We ask:

> Does coarse-graining commute with physical evolution?

In mathematical form, consider a microscopic state \(X\), microscopic
dynamics \(\Phi_{\mathrm{micro}}\), a coarse-graining map \(C\), and an
effective dynamical model \(\Phi_{\mathrm{eff}}\).

The two possible routes are

\[
X_0
\overset{\Phi_{\mathrm{micro}}}{\longrightarrow}
X(t)
\overset{C}{\longrightarrow}
C[X(t)]
\]

and

\[
X_0
\overset{C}{\longrightarrow}
C[X_0]
\overset{\Phi_{\mathrm{eff}}}{\longrightarrow}
\Phi_{\mathrm{eff}}[C[X_0]].
\]

The central quantity is therefore

\[
\boxed{
\Delta(t)
=
C[\Phi_{\mathrm{micro}}(t,X_0)]
-
\Phi_{\mathrm{eff}}(t,C[X_0])
}
\]

We call this the **scale-boundary discrepancy**.

---

## 1. The Basic Question

A physical system can often be described at several levels.

For example:

```text
microscopic particles
        ↓
kinetic / statistical description
        ↓
continuum fields
        ↓
macroscopic observables
