__all__ = ["CSVAutos", "CSVCross", "open_csv_autos", "open_csv_cross"]

# standard library
from dataclasses import dataclass, field

# dependencies
import numpy as np
import pandas as pd
import xarray as xr
import xarrayspecs as xs
from .common import AutoLSB, AutoUSB, Chan, Cross2SB, SpecVersion, Time
from ..utils import StrPath

# constants
COL_TIME = "time"
COL_FREQ = "freq[GHz]"
COL_USB = "out0"
COL_LSB = "out1"
COL_IMAG = "imag"
COL_REAL = "real"
TIME_DEFAULT = pd.to_datetime("2000")  # UTC
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"  # UTC


@dataclass
class CSVAutos(xs.AsDataset):
    """Data specification of DRS4 CSV (auto-correlations)."""

    # dims
    time: Time
    """Measured time in UTC."""

    chan: Chan
    """Channel number (0-511)."""

    # vars
    auto_usb: AutoUSB
    """Auto-correlation spectra of USB."""

    auto_lsb: AutoLSB
    """Auto-correlation spectra of LSB."""

    # attrs
    spec_version: SpecVersion = field(default=0, init=False)
    """Version of the data specification."""


@dataclass
class CSVCross(xs.AsDataset):
    """Data specification of DRS4 CSV (cross-correlation)."""

    # dims
    time: Time
    """Measured time in UTC."""

    chan: Chan
    """Channel number (0-511)."""

    # vars
    cross_2sb: Cross2SB
    """Cross-correlation spectra of 2SB (USB x LSB*)."""

    # attrs
    spec_version: SpecVersion = field(default=0, init=False)
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
        auto_lsb=ds[COL_LSB].data,
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
