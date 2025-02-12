__all__ = ["VDIF", "open_vdif"]


# standard library
from dataclasses import dataclass, field
from typing import Literal as L, Optional, get_args


# dependencies
import numpy as np
import xarray as xr
from numpy.typing import NDArray
from xarray_dataclasses import AsDataArray, Attr, Coordof, Data, Dataof
from .common import Chan, IntegTime, Time
from ..utils import StrPath, XarrayJoin


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
class Auto:
    data: Data[tuple[L["time"], L["chan"]], np.float64]
    long_name: Attr[str] = "Auto-correlation spectra"
    units: Attr[str] = "Arbitrary unit"


@dataclass
class VDIF(AsDataArray):
    """Data specification of DRS4 VDIF."""

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

    spec_version: Attr[int] = field(default=0, init=False)
    """Version of the data specification."""


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
    # for measurement (optional)
    integ_time: Optional[IntegTime] = None,
    # for file loading (optional)
    join: XarrayJoin = "inner",
) -> xr.DataArray:
    """Open a VDIF file as a DataArray.

    Args:
        vdif: Path of input VDIF file.
        integ_time: Spectral integration time in ms (100|200|500|1000).
            If not specified, it will be inferred from the VDIF file.
        join: Method of joining the first- and second-half spectra.

    Returns:
        DataArray of the input VDIF file.

    Raises:
        RuntimeError: Raised if the spectral integration time
            cannot be inferred from the VDIF file (frame number).
        ValueError: Raised if the given (or inferred) spectral
            integration time is other than 100|200|500|1000 ms.

    """
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
    seconds = word_0[0:30]
    frame_num = word_1[0:24]
    ref_epoch = word_1[24:30]

    if integ_time is None:
        integ_time = infer_integ_time(frame_num)

    if integ_time not in get_args(IntegTime):
        raise ValueError("Spectral integration time must be 100|200|500|1000.")

    time = (
        REF_EPOCH_ORIGIN
        + REF_EPOCH_UNIT * ref_epoch
        + np.timedelta64(1, "s") * seconds
        + np.timedelta64(integ_time, "ms") * (frame_num // 2)
    )

    return xr.concat(
        (
            VDIF.new(
                time=time[frame_num % 2 == 0],
                chan=CHAN_FIRST_HALF,
                auto=array["data"][frame_num % 2 == 0],
                integ_time=integ_time,
            ),
            VDIF.new(
                time=time[frame_num % 2 == 1],
                chan=CHAN_SECOND_HALF,
                auto=array["data"][frame_num % 2 == 1],
                integ_time=integ_time,
            ),
        ),
        dim="chan",
        join=join,
    )


def infer_integ_time(frame_num: NDArray[np.int_], /) -> IntegTime:
    """Infer spectral integration time from frame number."""
    if sum(frame_num == (frame_max := max(frame_num))) < 2:
        raise RuntimeError("Could not infer spectral integration time.")

    if (integ_time := 2000 // (frame_max + 1)) not in get_args(IntegTime):
        raise ValueError("Spectral integration time must be 100|200|500|1000.")

    return integ_time
