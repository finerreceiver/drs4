__all__ = ["Zarr", "open_csvs", "open_vdifs"]


# standard library
from dataclasses import dataclass, field
from typing import Literal as L, Optional, get_args


# dependencies
import numpy as np
import xarray as xr
from xarray_dataclasses import AsDataset, Attr, Coordof, Data, Dataof
from .common import (
    FREQ_INNER,
    FREQ_OUTER,
    Channel,
    Chassis,
    FreqRange,
    Interface,
    IntegTime,
    SideBand,
    Time,
    Chan,
    AutoUSB,
    AutoLSB,
    Cross2SB,
)
from .csv import open_csv_autos, open_csv_cross
from .vdif import open_vdif
from ..utils import StrPath, XarrayJoin


# data classes
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
class Zarr(AsDataset):
    """Data specification of DRS4 Zarr."""

    # dims
    time: Coordof[Time]
    """Measured time in UTC."""

    chan: Coordof[Chan]
    """Channel number (0-511)."""

    # coords
    freq: Coordof[Freq]
    """Intermediate frequency in GHz."""

    signal_sb: Coordof[SignalSB]
    """Signal sideband (USB|LSB|NA)."""

    signal_chan: Coordof[SignalChan]
    """Signal channel number (0-511|-1)."""

    # vars
    auto_usb: Dataof[AutoUSB]
    """Auto-correlation spectra of USB."""

    auto_lsb: Dataof[AutoLSB]
    """Auto-correlation spectra of LSB."""

    cross_2sb: Dataof[Cross2SB]
    """Cross-correlation spectra of 2SB (USB x LSB*)."""

    # attrs
    chassis: Attr[Chassis]
    """Chassis number of DRS4 (1|2)."""

    interface: Attr[Interface]
    """Interface (IF) number of DRS4 (1|2)."""

    integ_time: Attr[IntegTime]
    """Spectral integration time in ms (100|200|500|1000)."""

    spec_version: Attr[int] = field(default=0, init=False)
    """Version of the data specification."""


def open_csvs(
    csv_autos: StrPath,
    csv_cross: StrPath,
    /,
    *,
    # for measurement (required)
    chassis: Chassis,
    interface: Interface,
    freq_range: FreqRange,
    integ_time: IntegTime,
    # for measurement (optional)
    signal_sb: Optional[SideBand] = None,
    signal_chan: Optional[Channel] = None,
    # for file loading (optional)
    join: XarrayJoin = "inner",
) -> xr.Dataset:
    """Open CSV files of auto/cross correlations as a Dataset.

    Args:
        csv_autos: Path of input CSV file of auto-correlations.
        csv_cross: Path of input CSV file of cross-correlation.
        chassis: Chassis number of DRS4 (1|2).
        interface: Interface number of DRS4 (1|2).
        freq_range: Intermediate frequency range (inner|outer).
        integ_time: Spectral integration time in ms (100|200|500|1000).
        signal_sb: Signal sideband (USB|LSB).
            If not specified, NA (missing indicator) will be assigned.
        signal_chan: Signal channel number (0-511).
            If not specified, -1 (missing indicator) will be assigned.
        join: Method for joining the CSV files.

    Returns:
        Dataset of the input CSV files.

    Raises:
        ValueError: Raised if the given value of either chassis, freq_range,
            integ_time, or interface is not valid.

    """
    if chassis not in get_args(Chassis):
        raise ValueError("Chassis number must be 1|2.")

    if interface not in get_args(Interface):
        raise ValueError("Interface number must be 1|2.")

    if freq_range not in get_args(FreqRange):
        raise ValueError("Spectral integration time must be inner|outer.")

    if integ_time not in get_args(IntegTime):
        raise ValueError("Spectral integration time must be 100|200|500|1000.")

    ds_autos, ds_cross = xr.align(
        open_csv_autos(csv_autos),
        open_csv_cross(csv_cross),
        join=join,
    )

    return Zarr.new(
        # dims
        time=ds_autos.time.data,
        chan=ds_autos.chan.data,
        # coords
        freq=FREQ_INNER if freq_range == "inner" else FREQ_OUTER[::-1],
        signal_sb=np.full(
            ds_autos.sizes["time"],
            signal_sb if signal_sb is not None else "NA",
        ),
        signal_chan=np.full(
            ds_autos.sizes["time"],
            signal_chan if signal_chan is not None else -1,
        ),
        # vars
        auto_usb=ds_autos.auto_usb.data,
        auto_lsb=ds_autos.auto_lsb.data,
        cross_2sb=ds_cross.cross_2sb.data,
        # attrs
        chassis=chassis,
        interface=interface,
        integ_time=integ_time,
    )


def open_vdifs(
    vdif_usb: StrPath,
    vdif_lsb: StrPath,
    /,
    *,
    # for measurement (required)
    chassis: Chassis,
    interface: Interface,
    freq_range: FreqRange,
    # for measurement (optional)
    integ_time: Optional[IntegTime] = None,
    signal_sb: Optional[SideBand] = None,
    signal_chan: Optional[Channel] = None,
    # for file loading (optional)
    join: XarrayJoin = "inner",
) -> xr.Dataset:
    """Open USB/LSB VDIF files as a Dataset.

    Args:
        vdif_usb: Path of input USB VDIF file.
        vdif_lsb: Path of input LSB VDIF file.
        chassis: Chassis number of DRS4 (1|2).
        interface: Interface number of DRS4 (1|2).
        freq_range: Intermediate frequency range (inner|outer).
        integ_time: Spectral integration time in ms (100|200|500|1000).
            If not specified, it will be inferred from the VDIF files.
        signal_sb: Signal sideband (USB|LSB).
            If not specified, NA (missing indicator) will be assigned.
        signal_chan: Signal channel number (0-511).
            If not specified, -1 (missing indicator) will be assigned.
        join: Method for joining the VDIF files.

    Returns:
        Dataset of the input VDIF files.

    Raises:
        RuntimeError: Raised if USB/LSB spectral integration times are not same.
        ValueError: Raised if the given value of either chassis, freq_range,
            integ_time, or interface is not valid.

    """
    if chassis not in get_args(Chassis):
        raise ValueError("Chassis number must be 1|2.")

    if interface not in get_args(Interface):
        raise ValueError("Interface number must be 1|2.")

    if freq_range not in get_args(FreqRange):
        raise ValueError("Spectral integration time must be inner|outer.")

    da_usb, da_lsb = xr.align(
        open_vdif(vdif_usb, integ_time=integ_time, join=join),
        open_vdif(vdif_lsb, integ_time=integ_time, join=join),
        join=join,
    )

    if da_usb.integ_time != da_lsb.integ_time:
        raise RuntimeError("USB/LSB spectral integration times must be same.")

    return Zarr.new(
        # dims
        time=da_usb.time.data,
        chan=da_usb.chan.data,
        # coords
        freq=FREQ_INNER if freq_range == "inner" else FREQ_OUTER[::-1],
        signal_sb=np.full(
            da_usb.sizes["time"],
            signal_sb if signal_sb is not None else "NA",
        ),
        signal_chan=np.full(
            da_usb.sizes["time"],
            signal_chan if signal_chan is not None else -1,
        ),
        # vars
        auto_usb=da_usb.data,
        auto_lsb=da_lsb.data,
        cross_2sb=np.full(da_usb.shape, np.nan),
        # attrs
        chassis=chassis,
        interface=interface,
        integ_time=da_usb.integ_time,
    )
