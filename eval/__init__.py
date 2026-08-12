"""
NFL Vision App - Evaluation Package
Metrics and evaluation harness.

Note: Imports are lazy to avoid loading heavy dependencies on init.
"""

__all__ = ['Evaluator']

def __getattr__(name):
    """Lazy load modules on first access."""
    if name == 'Evaluator':
        from .metrics import Evaluator
        return Evaluator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
