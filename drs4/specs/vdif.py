__all__ = ["VDIF", "open_vdif"]


# standard library
from dataclasses import dataclass, field
from os import PathLike
from typing import Literal as L, Union, get_args


# dependencies
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from xarray_dataclasses import AsDataArray, Attr, Coordof, Data, Dataof


# type hints
IntegTime = L[100, 200, 500, 1000]
Join = L["outer", "inner", "left", "right", "exact", "override"]
StrPath = Union[PathLike[str], str]


# constants
CHAN_FIRST_HALF = np.arange(0, 256)
CHAN_SECOND_HALF = np.arange(256, 512)
REF_EPOCH_ORIGIN = np.datetime64("2000", "Y")  # UTC
REF_EPOCH_UNIT = np.timedelta64(6, "M")
VDIF_HEADER_BYTES = 32
VDIF_DATA_BYTES = 1024
VDIF_FRAME_BYTES = VDIF_HEADER_BYTES + VDIF_DATA_BYTES


# data classes
@dataclass
class Time:
    data: Data[L["time"], L["M8[ns]"]]
    long_name: Attr[str] = "Measured time in UTC"


@dataclass
class Chan:
    data: Data[L["chan"], np.int64]
    long_name: Attr[str] = "Channel number"


@dataclass
class Auto:
    data: Data[tuple[L["time"], L["chan"]], np.float64]
    long_name: Attr[str] = "Auto-correlation spectra"
    units: Attr[str] = "Arbitrary unit"


@dataclass
class VDIF(AsDataArray):
    """Data specifications of DRS4 VDIF."""

    # dims
    time: Coordof[Time]
    """Measured time in UTC."""

    chan: Coordof[Chan]
    """Channel number (0-511)."""

    # vars
    auto: Dataof[Auto]
    """Auto-correlation spectra."""

    # attrs
    integ_time: Attr[IntegTime]
    """Spectral integration time in ms (100|200|500|1000)."""

    version: Attr[int] = field(default=0, init=False)
    """Version of the data specifications."""


@dataclass(frozen=True)
class Word:
    """VDIF header word parser."""

    data: NDArray[np.int_]
    """VDIF header word as a 1D integer array."""

    def __getitem__(self, index: slice, /) -> NDArray[np.int_]:
        """Slice the VDIF header word."""
        start, stop = index.start, index.stop
        return (self.data >> start) & ((1 << stop - start) - 1)


def open_vdif(
    vdif: StrPath,
    /,
    *,
    integ_time: IntegTime = 100,
    join: Join = "inner",
) -> xr.DataArray:
    """Open a VDIF file as a DataArray.

    Args:
        vdif: Path of input VDIF file.
        integ_time: Spectral integration time in ms.
        join: Method of joining the first and second-half spectra.

    Returns:
        DataArray of the input VDIF file.

    """
    if integ_time not in get_args(IntegTime):
        raise ValueError("Value of integ_time must be 100|200|500|1000.")

    array = np.fromfile(
        vdif,
        dtype=[
            ("word_0", "u4"),
            ("word_1", "u4"),
            ("word_2", "u4"),
            ("word_3", "u4"),
            ("word_4", "u4"),
            ("word_5", "u4"),
            ("word_6", "u4"),
            ("word_7", "u4"),
            ("data", ("f4", 256)),
        ],
    )
    word_0 = Word(array["word_0"])
    word_1 = Word(array["word_1"])
    is_first_half = word_1[0:24] % 2 == 0
    is_second_half = word_1[0:24] % 2 == 1

    time = (
        REF_EPOCH_ORIGIN
        + REF_EPOCH_UNIT * word_1[24:30]
        + np.timedelta64(1, "s") * word_0[0:30]
        + np.timedelta64(integ_time, "ms") * (word_1[0:24] // 2)
    )

    return xr.concat(
        (
            VDIF.new(
                time=time[is_first_half],
                chan=CHAN_FIRST_HALF,
                auto=array["data"][is_first_half],
                integ_time=integ_time,
            ),
            VDIF.new(
                time=time[is_second_half],
                chan=CHAN_SECOND_HALF,
                auto=array["data"][is_second_half],
                integ_time=integ_time,
            ),
        ),
        dim="chan",
        join=join,
    )
