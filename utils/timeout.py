import sys
import time
from functools import wraps

# Signal-based timeout only available on Unix
_HAS_SIGALRM = hasattr(__import__('signal'), 'SIGALRM')
if _HAS_SIGALRM:
    import signal


class TimeoutException(Exception):
    """Raised when schedule exceeds the wall-clock timeout."""
    pass


class DeferredTimeout:
    """Context manager for deferred timeout handling.

    Instead of raising immediately when timeout fires, sets a flag.
    User code should check `timed_out` after interruptible operations.

    On Unix: Uses SIGALRM for immediate flag setting.
    On Windows: Uses elapsed time check (flag updated on property access).
    """
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self._timed_out = False
        self._start_time = None
        self._old_handler = None

    @property
    def timed_out(self):
        """Check if timeout has expired."""
        if self._timed_out:
            return True
        if self.timeout_seconds is not None and self._start_time is not None:
            if time.time() - self._start_time >= self.timeout_seconds:
                self._timed_out = True
        return self._timed_out

    def __enter__(self):
        if self.timeout_seconds is None:
            return self

        self._start_time = time.time()

        if _HAS_SIGALRM:
            def _handler(signum, frame):
                self._timed_out = True

            self._old_handler = signal.signal(signal.SIGALRM, _handler)
            signal.alarm(int(self.timeout_seconds))

        return self

    def __exit__(self, *args):
        if self.timeout_seconds is not None and _HAS_SIGALRM:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old_handler)


def timeout_from_kwarg(timeout_kw="timeout", exc=TimeoutException):
    """
    Decorator that:
      - reads timeout from a keyword argument (e.g., 'timeout')
      - installs a SIGALRM handler (Unix) or uses threading timer (Windows)
      - raises `exc` when the timeout expires
    On Unix: Uses SIGALRM for immediate interruption.
    On Windows: Uses threading.Timer (may not interrupt blocking operations).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timeout = kwargs.get(timeout_kw, None)
            # If no timeout is given, just run normally
            if timeout is None:
                return func(*args, **kwargs)

            if _HAS_SIGALRM:
                # Unix: use SIGALRM
                def _handler(signum, frame):
                    raise exc()

                old_handler = signal.signal(signal.SIGALRM, _handler)
                signal.alarm(int(timeout))

                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            else:
                # Windows: use threading timer (less reliable for blocking ops)
                import threading
                timer_fired = [False]

                def _timeout_handler():
                    timer_fired[0] = True

                timer = threading.Timer(timeout, _timeout_handler)
                timer.start()

                try:
                    result = func(*args, **kwargs)
                    if timer_fired[0]:
                        raise exc()
                    return result
                finally:
                    timer.cancel()

        return wrapper
    return decorator