from __future__ import annotations

import time
from functools import wraps

from torchbci.block.base.base_block import BaseBlock


class CombinatoBlock(BaseBlock):
    """Base class for Combinato pipeline blocks with automatic runtime timing."""

    def __init__(self):
        super().__init__()
        self._measure_runtime = True
        self._last_runtime = None

    @staticmethod
    def measure_runtime_decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if not getattr(self, "_measure_runtime", False):
                return func(self, *args, **kwargs)
            start = time.perf_counter()
            out = func(self, *args, **kwargs)
            self._last_runtime = time.perf_counter() - start
            return out
        return wrapper

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "forward" in cls.__dict__:
            cls.forward = CombinatoBlock.measure_runtime_decorator(cls.forward)

    def enable_runtime_measure(self, enable=True):
        self._measure_runtime = enable

    def runtime_measure(self):
        return self._last_runtime
