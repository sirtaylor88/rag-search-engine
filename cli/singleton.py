"""Generic singleton base class."""

from typing import Any, ClassVar, TypeVar, cast

_T = TypeVar("_T", bound="Singleton")


class Singleton:
    """Base class that ensures only one instance exists per subclass.

    Subclasses must guard ``__init__`` with ``if self._initialized: return``
    at the top and set ``self._initialized = True`` at the end of first-time
    setup, so re-entrant calls are no-ops.
    """

    _instance: ClassVar["Singleton | None"] = None
    _initialized: bool = False

    def __new__(cls: type[_T], *_args: Any, **_kwargs: Any) -> _T:
        """Return the singleton instance, creating it on first call.

        Args:
            cls (type[_T]): The class being instantiated.

        Returns:
            _T: The shared singleton instance.
        """
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instance = instance
        return cast(_T, cls._instance)
