# ABOUTME: Pass@K callback subpackage: decision engine, runners, and the stopping callback.
# ABOUTME: The callback is exported lazily so importing decisions alone stays free of the eval stack.

__all__ = ["PassAtKStoppingCallback"]


def __getattr__(name):
    if name == "PassAtKStoppingCallback":
        from .callback import PassAtKStoppingCallback
        return PassAtKStoppingCallback
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
