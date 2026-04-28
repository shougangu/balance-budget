# ABOUTME: Backwards-compatible re-export. Production code imports
# ABOUTME: PassAtKStoppingCallback from this path; the implementation lives in passk/.

from tuning.training.passk.callback import PassAtKStoppingCallback

__all__ = ["PassAtKStoppingCallback"]
