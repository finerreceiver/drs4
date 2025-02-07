__all__ = ["Zarr", "open_vdifs"]


# standard library
from dataclasses import dataclass, field
from os import PathLike
from typing import Literal as L, Union, get_args


# dependencies
import numpy as np
import xarray as xr
from xarray_dataclasses import AsDataset, Attr, Coordof, Data, Dataof
from .vdif import open_vdif


# type hints
Chassis = L[1, 2]
FreqRange = L["inner", "outer"]
Interface = L[1, 2]
IntegTime = L[100, 200, 500, 1000]
SideBand = L["USB", "LSB", "NA"]
StrPath = Union[PathLike[str], str]


# constants
FREQ_INTERVAL = 0.02  # GHz
FREQ_INNER = FREQ_INTERVAL * np.arange(0, 512)  # GHz
FREQ_OUTER = FREQ_INTERVAL * np.arange(512, 1024)[::-1]  # GHz


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
    """Signal sideband (USB|LSB|NA)."""

    # vars
    auto_USB: Dataof[AutoUSB]
    """Auto-correlation spectra of USB."""

    auto_LSB: Dataof[AutoLSB]
    """Auto-correlation spectra of LSB."""

    cross_2SB: Dataof[Cross2SB]
    """Cross-correlation spectra of 2SB (USB x LSB*)."""

    # attrs
    chassis: Attr[Chassis]
    """Chassis number of DRS4 (1|2)."""

    freq_range: Attr[FreqRange]
    """Intermediate frequency range (inner|outer)."""

    integ_time: Attr[IntegTime]
    """Spectral integration time in ms (100|200|500|1000)."""

    interface: Attr[Interface]
    """Interface (IF) number of DRS4 (1|2)."""

    version: Attr[int] = field(default=0, init=False)
    """Version of the data specifications."""


def open_vdifs(
    vdif_usb: StrPath,
    vdif_lsb: StrPath,
    /,
    *,
    chassis: Chassis = 1,
    freq_range: FreqRange = "inner",
    integ_time: IntegTime = 100,
    interface: Interface = 1,
    signal_chan: int = 0,
    signal_SB: SideBand = "NA",
) -> xr.Dataset:
    """Open USB/LSB VDIF files as a Dataset.

    Args:
        vdif_usb: Path of input USB VDIF file.
        vdif_lsb: Path of input LSB VDIF file.
        chassis: Chassis number of DRS4.
        freq_range: Intermediate frequency range.
        integ_time: Spectral integration time in ms.
        interface: Interface number of DRS4.
        signal_chan: Signal channel number.
        signal_SB: Signal sideband.

    Returns:
        Dataset of the input VDIF files.

    """
    if chassis not in get_args(Chassis):
        raise ValueError("Value of chassis must be 1|2.")

    if freq_range not in get_args(FreqRange):
        raise ValueError("Value of freq_range must be inner|outer.")

    if integ_time not in get_args(IntegTime):
        raise ValueError("Value of integ_time must be 100|200|500|1000.")

    if interface not in get_args(Interface):
        raise ValueError("Value of interface must be 1|2.")

    da_usb, da_lsb = xr.align(
        open_vdif(vdif_usb, integ_time=integ_time),
        open_vdif(vdif_lsb, integ_time=integ_time),
        join="inner",
    )

    return Zarr.new(
        # dims
        time=da_usb.time.data,
        chan=da_usb.chan.data,
        # coords
        freq=FREQ_INNER if freq_range == "inner" else FREQ_OUTER,
        signal_chan=np.full(da_usb.shape[0], signal_chan),
        signal_SB=np.full(da_usb.shape[0], signal_SB),
        # vars
        auto_USB=da_usb.data,
        auto_LSB=da_lsb.data,
        cross_2SB=np.full(da_usb.shape, np.nan),
        # attrs
        chassis=chassis,
        freq_range=freq_range,
        integ_time=integ_time,
        interface=interface,
    )
