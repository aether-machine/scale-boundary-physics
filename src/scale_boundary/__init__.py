"""
Scale Boundary Physics

A computational framework for investigating whether coarse-graining,
dynamical evolution, and changes of physical description commute.

The central object of study is the discrepancy

    Delta(t) = C[Phi_micro(t, X0)]
              - Phi_eff(t, C[X0])

where:

    X0          microscopic initial state
    Phi_micro   microscopic dynamics
    C           coarse-graining / representation map
    Phi_eff     effective dynamics

The project begins with controlled toy systems and is intended to
progress toward kinetic and continuum descriptions.
"""

__version__ = "0.1.0"

__all__ = [
    "__version__",
]
