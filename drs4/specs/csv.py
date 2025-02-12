__all__ = ["CSVAutos", "CSVCross", "open_csv_autos", "open_csv_cross"]


# standard library
from dataclasses import dataclass, field
from os import PathLike
from typing import Literal as L, Union


# dependencies
import numpy as np
import pandas as pd
import xarray as xr
from xarray_dataclasses import AsDataset, Attr, Coordof, Data, Dataof


# type hints
StrPath = Union[PathLike[str], str]


# constants
COL_TIME = "time"
COL_FREQ = "freq[GHz]"
COL_USB = "out0"
COL_LSB = "out1"
COL_IMAG = "imag"
COL_REAL = "real"
TIME_DEFAULT = pd.to_datetime("2000")  # UTC
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"  # UTC


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
class CSVAutos(AsDataset):
    """Data specification of DRS4 CSV (auto-correlations)."""

    # dims
    time: Coordof[Time]
    """Measured time in UTC."""

    chan: Coordof[Chan]
    """Channel number (0-511)."""

    # vars
    auto_usb: Dataof[AutoUSB]
    """Auto-correlation spectra of USB."""

    auto_lsb: Dataof[AutoLSB]
    """Auto-correlation spectra of LSB."""

    # attrs
    spec_version: Attr[int] = field(default=0, init=False)
    """Version of the data specification."""


@dataclass
class CSVCross(AsDataset):
    """Data specification of DRS4 CSV (cross-correlation)."""

    # dims
    time: Coordof[Time]
    """Measured time in UTC."""

    chan: Coordof[Chan]
    """Channel number (0-511)."""

    # vars
    cross_2sb: Dataof[Cross2SB]
    """Cross-correlation spectra of 2SB (USB x LSB*)."""

    # attrs
    spec_version: Attr[int] = field(default=0, init=False)
    """Version of the data specification."""


def open_csv_autos(csv: StrPath, /) -> xr.Dataset:
    """Open a CSV file of auto-correlations as a Dataset.

    Args:
        csv: Path of input CSV file (e.g. new_pow.csv).

    Returns:
        Dataset of the input CSV file.

    """
    try:
        df = pd.read_csv(csv, parse_dates=[COL_TIME])
    except ValueError:
        df = pd.read_csv(csv).assign(time=TIME_DEFAULT)

    ds = df.set_index([COL_TIME, COL_FREQ]).to_xarray()

    return CSVAutos.new(
        time=ds[COL_TIME].data,
        chan=np.arange(ds.sizes[COL_FREQ]),
        auto_usb=ds[COL_USB].data,
        auto_lsb=ds[COL_USB].data,
    )


def open_csv_cross(csv: StrPath, /) -> xr.Dataset:
    """Open a CSV file of cross-correlation as a Dataset.

    Args:
        csv: Path of input CSV file (e.g. new_phase.csv).

    Returns:
        Dataset of the input CSV file.

    """
    try:
        df = pd.read_csv(csv, parse_dates=[COL_TIME])
    except ValueError:
        df = pd.read_csv(csv).assign(time=TIME_DEFAULT)

    ds = df.set_index([COL_TIME, COL_FREQ]).to_xarray()

    return CSVCross.new(
        time=ds[COL_TIME].data,
        chan=np.arange(ds.sizes[COL_FREQ]),
        cross_2sb=(ds[COL_REAL] + ds[COL_IMAG] * 1j).data,
    )
