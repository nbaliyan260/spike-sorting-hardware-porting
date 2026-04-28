import time
import torch
from abc import abstractmethod
from functools import wraps

class Block(torch.nn.Module):
    """Base class for all blocks in torchbci."""
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
            print(f"[{self.__class__.__name__}] Runtime (ms): {self._last_runtime * 1000:.3f}")
            return out
        return wrapper

    def __init_subclass__(cls):
        super().__init_subclass__()
        if hasattr(cls, "forward"):
            cls.forward = cls.measure_runtime_decorator(cls.forward)

    def enable_runtime_measure(self, enable=True):
        """Turn runtime measurement on or off."""
        self._measure_runtime = enable

    def runtime_measure(self):
        """Return the last measured runtime in seconds."""
        return self._last_runtime

    @abstractmethod
    def visualize(self):
        pass