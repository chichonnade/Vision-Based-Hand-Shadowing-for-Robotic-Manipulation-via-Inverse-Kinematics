"""Data-transformation classes."""
# pylint: disable=arguments-differ
import abc
import time
from typing import Generic, TypeVar

InT = TypeVar("InT")
OutT = TypeVar("OutT")


class Transformation(abc.ABC, Generic[InT, OutT]):
    """Abstract base class for a single pipeline stage.

    Sub-classes implement _transform.  The public __call__ method
    wraps it, tracks extra information like latency, and returns the result.
    """

    def __init__(self) -> None:
        self._latency_ms: float | None = None

    def __call__(self, data_in: InT, /) -> OutT:
        """Execute the transformation."""
        start = time.perf_counter()
        result: OutT = self._transform(data_in)
        self._latency_ms = (time.perf_counter() - start) * 1000
        return result

    @property
    def latency_ms(self) -> float | None:
        """Milliseconds spent inside the most recent call."""
        return self._latency_ms

    @abc.abstractmethod
    def _transform(self, data_in: InT, /) -> OutT:
        """Transform data."""
        raise NotImplementedError
