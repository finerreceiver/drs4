__all__ = ["StrPath", "XarrayJoin", "is_strpath", "set_logger", "set_workdir", "unique"]


# standard library
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from logging import FileHandler, StreamHandler, getLogger
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal as L, Optional, Union


# dependencies
import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypeGuard


# constants
LOGGER = getLogger(__name__)


# type hints
Axis = Optional[Union[Sequence[int], int]]
StrPath = Union[PathLike[str], str]
XarrayJoin = L["outer", "inner", "left", "right", "exact", "override"]


def is_strpath(obj: Any, /) -> TypeGuard[StrPath]:
    """Check if given object can provide a file system path."""
    return isinstance(obj, (PathLike, str))


@contextmanager
def set_workdir(workdir: Optional[StrPath] = None, /) -> Iterator[Path]:
    """Set the working directory for output VDIF files."""
    if workdir is not None:
        yield Path(workdir).expanduser()
    else:
        with TemporaryDirectory() as workdir:
            yield Path(workdir)


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


def set_logger(
    *,
    file: StrPath = "drs4.log",
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

    if file:
        logger.addHandler(FileHandler(file))

    if stderr:
        logger.addHandler(StreamHandler())
