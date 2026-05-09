"""Tests for cli.singleton."""

import pytest

from cli.singleton import Singleton


class _ConcreteA(Singleton):
    def __init__(self) -> None:
        if self._initialized:
            return
        self.value = 42
        self._initialized = True


class _ConcreteB(Singleton):
    def __init__(self) -> None:
        if self._initialized:
            return
        self.value = 99
        self._initialized = True


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset both concrete singletons before each test."""
    monkeypatch.setattr(_ConcreteA, "_instance", None)
    monkeypatch.setattr(_ConcreteB, "_instance", None)


def test_same_instance_returned_on_multiple_calls() -> None:
    """Two calls to the same subclass should return the identical object."""
    assert _ConcreteA() is _ConcreteA()


def test_init_runs_only_once() -> None:
    """Mutating the instance between calls should persist; __init__ must not reset."""
    a = _ConcreteA()
    a.value = 100
    assert _ConcreteA().value == 100


def test_different_subclasses_have_independent_instances() -> None:
    """Each subclass must maintain its own independent singleton instance."""
    assert _ConcreteA() is not _ConcreteB()


def test_initialized_flag_true_after_first_call() -> None:
    """_initialized must be True after the first instantiation."""
    a = _ConcreteA()
    assert a._initialized is True  # pylint: disable=protected-access
