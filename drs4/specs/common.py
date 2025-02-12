__all__ = [
    # type hints
    "Channel",
    "Chassis",
    "FreqRange",
    "Interface",
    "IntegTime",
    "SideBand",
    # constants
    "CHAN_TOTAL",
    "FREQ_INTERVAL",
    "FREQ_INNER",
    "FREQ_OUTER",
    "OBSID_FORMAT",
    "ZARR_FORMAT",
    # data classes
    "Time",
    "Chan",
    "AutoUSB",
    "AutoLSB",
    "Cross2SB",
]


# standard library
from dataclasses import dataclass
from typing import Literal as L


# dependencies
import numpy as np
from xarray_dataclasses import Attr, Data


# type hints
Channel = int
Chassis = L[1, 2]
FreqRange = L["inner", "outer"]
Interface = L[1, 2]
IntegTime = L[100, 200, 500, 1000]  # ms
SideBand = L["USB", "LSB"]


# constants
CHAN_TOTAL = 512  # ch
FREQ_INTERVAL = 0.02  # GHz
FREQ_INNER = FREQ_INTERVAL * np.arange(CHAN_TOTAL * 0, CHAN_TOTAL * 1)  # GHz
FREQ_OUTER = FREQ_INTERVAL * np.arange(CHAN_TOTAL * 1, CHAN_TOTAL * 2)  # GHz
OBSID_FORMAT = "%Y%m%dT%H%M%SZ"
ZARR_FORMAT = "drs4-{0}-chassis{1}-if{2}.zarr.zip"


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
