__all__ = [
    # type hints
    "Channel",
    "Chassis",
    "FreqRange",
    "Interface",
    "IntegTime",
    "SideBand",
    # constants (data formats)
    "CHAN_TOTAL",
    "FREQ_INTERVAL",
    "FREQ_INNER",
    "FREQ_OUTER",
    # constants (file formats)
    "CSV_AUTOS_FORMAT",
    "CSV_CROSS_FORMAT",
    "OBSID_FORMAT",
    "VDIF_FORMAT",
    "ZARR_FORMAT",
    # constants (environment variables)
    "ENV_CTRL_ADDR",
    "ENV_CTRL_USER",
    "ENV_DEST_ADDR",
    "ENV_DEST_PORT1",
    "ENV_DEST_PORT2",
    "ENV_DEST_PORT3",
    "ENV_DEST_PORT4",
    "ENV_LO_FREQ",
    "ENV_LO_MULT",
    "ENV_SG_ADDR",
    "ENV_SG_AMPL",
    "ENV_SG_PORT",
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


# constants (data formats)
CHAN_TOTAL = 512  # ch
FREQ_INTERVAL = 0.02  # GHz
FREQ_INNER = FREQ_INTERVAL * np.arange(CHAN_TOTAL * 0, CHAN_TOTAL * 1)  # GHz
FREQ_OUTER = FREQ_INTERVAL * np.arange(CHAN_TOTAL * 1, CHAN_TOTAL * 2)  # GHz


# constants (file formats)
CSV_AUTOS_FORMAT = "drs4-{0}-chassis{1}-autos-if{2}.csv"
CSV_CROSS_FORMAT = "drs4-{0}-chassis{1}-cross-if{2}.csv"
OBSID_FORMAT = "%Y%m%dT%H%M%SZ"
VDIF_FORMAT = "drs4-{0}-chassis{1}-in{2}.vdif"
ZARR_FORMAT = "drs4-{0}-chassis{1}-if{2}.zarr.zip"


# constants (environment variavles)
ENV_CTRL_ADDR = "DRS4_CHASSIS{0}_CTRL_ADDR"
ENV_CTRL_USER = "DRS4_CHASSIS{0}_CTRL_USER"
ENV_DEST_ADDR = "DRS4_CHASSIS{0}_DEST_ADDR"
ENV_DEST_PORT1 = "DRS4_CHASSIS{0}_DEST_PORT1"
ENV_DEST_PORT2 = "DRS4_CHASSIS{0}_DEST_PORT2"
ENV_DEST_PORT3 = "DRS4_CHASSIS{0}_DEST_PORT3"
ENV_DEST_PORT4 = "DRS4_CHASSIS{0}_DEST_PORT4"
ENV_LO_FREQ = "DRS4_LO_FREQ"
ENV_LO_MULT = "DRS4_LO_MULT"
ENV_SG_ADDR = "DRS4_CW_SG_ADDR"
ENV_SG_AMPL = "DRS4_CW_SG_AMPL"
ENV_SG_PORT = "DRS4_CW_SG_PORT"


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
