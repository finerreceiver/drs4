__all__ = ["Zarr"]


# standard library
from dataclasses import dataclass
from typing import Literal as L


# dependencies
import numpy as np
from xarray_dataclasses import AsDataset, Attr, Coordof, Data, Dataof


# constants
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
class Freq:
    data: Data[L["chan"], np.float64]
    long_name: Attr[str] = "Intermediate frequency"
    units: Attr[str] = "GHz"


@dataclass
class SignalChan:
    data: Data[L["time"], np.int64]
    long_name: Attr[str] = "Signal channel number"


@dataclass
class SignalSB:
    data: Data[L["time"], L["U3"]]
    long_name: Attr[str] = "Signal sideband"


@dataclass
class AutoUSB:
    data: Data[tuple[L["time"], L["chan"]], np.float64]
    long_name: Attr[str] = "Auto-correlation spectra of USB"
    units: Attr[str] = "Arbitrary unit"


@dataclass
class AutoLSB:
    data: Data[tuple[L["time"], L["chan"]], np.float64]
    long_name: Attr[str] = "Auto-correlation spectra of LSB"
    units: Attr[str] = "Arbitrary unit"


@dataclass
class Cross2SB:
    data: Data[tuple[L["time"], L["chan"]], np.complex128]
    long_name: Attr[str] = "Cross-correlation spectra of 2SB"
    units: Attr[str] = "Arbitrary unit"


@dataclass
class Zarr(AsDataset):
    """Data specifications of DRS4 Zarr."""

    # dims
    time: Coordof[Time]
    """Measured time in UTC."""

    chan: Coordof[Chan]
    """Channel number (0-511)."""

    # coords
    freq: Coordof[Freq]
    """Intermediate frequency in GHz."""

    signal_chan: Coordof[SignalChan]
    """Signal channel number (0-511)."""

    signal_SB: Coordof[SignalSB]
    """Signal sideband (USB|LSB)."""

    # vars
    auto_USB: Dataof[AutoUSB]
    """Auto-correlation spectra of USB."""

    auto_LSB: Dataof[AutoLSB]
    """Auto-correlation spectra of LSB."""

    cross_2SB: Dataof[Cross2SB]
    """Cross-correlation spectra of 2SB (USB x LSB*)."""

    # attrs
    chassis: Attr[L[1, 2]]
    """Chassis number of DRS4 (1|2)."""

    interface: Attr[L[1, 2]]
    """Interface (IF) number of DRS4 (1|2)."""

    integ_time: Attr[L[100, 200, 500, 1000]]
    """Spectral integration time in ms (100|200|500|1000)."""
