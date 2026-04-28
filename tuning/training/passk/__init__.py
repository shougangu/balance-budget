# ABOUTME: Pass@K callback subpackage — split from the previous monolithic passk_callback.py.
# ABOUTME: Public re-export below preserves the historical import path.

from .callback import PassAtKStoppingCallback

__all__ = ["PassAtKStoppingCallback"]
