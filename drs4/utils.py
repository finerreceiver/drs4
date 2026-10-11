__all__ = [
    "StrPath",
    "XarrayJoin",
    "is_strpath",
    "log_args",
    "set_logger",
    "set_workdir",
    "unique",
]

# standard library
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from functools import wraps
from inspect import signature
from logging import FileHandler, Formatter, Logger, StreamHandler, getLogger
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import gmtime
from typing import Any, Literal as L, TypeVar

# dependencies
import numpy as np
from numpy.typing import NDArray
from typing_extensions import ParamSpec, TypeGuard

# constants
LOGGER_LOCK = Lock()
LOGGER = getLogger(__name__)

# type hints
P = ParamSpec("P")
R = TypeVar("R")
Axis = Sequence[int] | int | None
StrPath = PathLike[str] | str
XarrayJoin = L["outer", "inner", "left", "right", "exact", "override"]


def is_strpath(obj: Any, /) -> TypeGuard[StrPath]:
    """Check if given object can provide a file system path."""
    return isinstance(obj, (PathLike, str))


def log_args(logger: Logger, /) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to log function arguments."""

    def decorator(func: Callable[P, R], /) -> Callable[P, R]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            sig = signature(func)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()

            with LOGGER_LOCK:
                logger.debug("(")

                for key, val in bound.arguments.items():
                    logger.debug(f"  {key}: {val!r}")

                logger.debug(")")

            return func(*args, **kwargs)

        return wrapper

    return decorator


@contextmanager
def set_workdir(workdir: StrPath | None = None, /) -> Iterator[Path]:
    """Set the working directory for output VDIF files."""
    if workdir is not None:
        yield Path(workdir).expanduser()
    else:
        with TemporaryDirectory() as workdir:
            yield Path(workdir)


def set_logger(
    *,
    file: StrPath | None = None,
    level: str = "INFO",
    stderr: bool = True,
) -> None:
    """Setup the DRS4 package root logger.

    Args:
        file: If given, log will be written in it.
        level: Logging level (e.g., INFO, DEBUG).
        stderr: If True, log will be sent to stderr.

    """
    logger = getLogger(__name__.split(".")[0])
    logger.setLevel(level.upper())
    logger.handlers = []

    formatter = Formatter(
        fmt="[{asctime}Z {name}.{funcName} ({threadName}) {levelname}] {message}",
        datefmt="%Y-%m-%dT%H:%M:%S",
        style="{",
    )
    formatter.converter = gmtime

    if file:
        handler = FileHandler(file)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if stderr:
        handler = StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def unique(array: NDArray[Any], /, axis: Axis = None) -> NDArray[Any]:
    """Return unique values along given axis (axes)."""
    if axis is None:
        axis = list(range(array.ndim))

    axes = np.atleast_1d(axis)
    shape = np.array(array.shape)
    newshape = np.prod(shape[axes]), *np.delete(shape, axes)

    for ax in axes:
        array = np.moveaxis(array, ax, 0)

    result = np.unique(array.reshape(newshape), axis=0)

    if result.shape[0] != 1:
        raise ValueError("Array values are not unique.")

    return result[0]
