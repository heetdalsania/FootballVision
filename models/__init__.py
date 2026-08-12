"""
NFL Vision App - Models Package
Formation classification models.

Note: Imports are lazy to avoid loading heavy dependencies on init.
"""

__all__ = ['FormationClassifier', 'Formation']

def __getattr__(name):
    """Lazy load modules on first access."""
    if name == 'FormationClassifier':
        from .classifier import FormationClassifier
        return FormationClassifier
    elif name == 'Formation':
        from .classifier import Formation
        return Formation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
