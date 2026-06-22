__all__ = [
    # type hints (attr)
    "Channel",
    "Chassis",
    "DSPMode",
    "FreqRange",
    "Interface",
    "IntegTime",
    "SideBand",
    "SpecVersion",
    # type hints (coord/data)
    "Time",
    "Chan",
    "AutoUSB",
    "AutoLSB",
    "Cross2SB",
    # constants (data format)
    "CHAN_TOTAL",
    "FREQ_INTERVAL",
    "FREQ_INNER",
    "FREQ_OUTER",
    # constants (file format)
    "CSV_AUTOS_FORMAT",
    "CSV_CROSS_FORMAT",
    "OBSID_FORMAT",
    "VDIF_FORMAT",
    "ZARR_CHUNKS",
    "ZARR_ENCODING",
    "ZARR_FORMAT",
    "ZARR_SHARDS",
    # constants (environment variable)
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
]

# standard library
from typing import Annotated, Literal as L

# dependencies
import numpy as np
import xarrayspecs as xs

# type hints (attr)
Channel = xs.Attr[int]
Chassis = xs.Attr[L[1, 2]]
DSPMode = xs.Attr[L["IQ", "SB"]]
FreqRange = xs.Attr[L["inner", "outer"]]
Interface = xs.Attr[L[1, 2]]
IntegTime = xs.Attr[L[100, 200, 500, 1000]]  # ms
SideBand = xs.Attr[L["USB", "LSB"]]
SpecVersion = xs.Attr[int]

# type hints (coord/data)
Time = Annotated[
    xs.Coord[L["time"], L["M8[ns]"]],
    xs.attrs(long_name="Measured time in UTC"),
]
Chan = Annotated[
    xs.Coord[L["chan"], np.int64],
    xs.attrs(long_name="Channel number"),
]
AutoLSB = Annotated[
    xs.Data[tuple[L["time"], L["chan"]], np.float64],
    xs.attrs(
        long_name="Auto-correlation spectra of LSB",
        units="Arbitrary unit",
    ),
]
AutoUSB = Annotated[
    xs.Data[tuple[L["time"], L["chan"]], np.float64],
    xs.attrs(
        long_name="Auto-correlation spectra of USB",
        units="Arbitrary unit",
    ),
]
Cross2SB = Annotated[
    xs.Data[tuple[L["time"], L["chan"]], np.complex128],
    xs.attrs(
        long_name="Cross-correlation spectra of 2SB",
        units="Arbitrary unit",
    ),
]

# constants (data format)
CHAN_TOTAL = 512  # ch
FREQ_INTERVAL = 0.02  # GHz
FREQ_INNER = FREQ_INTERVAL * np.arange(CHAN_TOTAL * 0, CHAN_TOTAL * 1)  # GHz
FREQ_OUTER = FREQ_INTERVAL * (np.arange(CHAN_TOTAL * 1, CHAN_TOTAL * 2) + 1)  # GHz

# constants (file format)
CSV_AUTOS_FORMAT = "drs4-{0}-chassis{1}-autos-if{2}.csv"
CSV_CROSS_FORMAT = "drs4-{0}-chassis{1}-cross-if{2}.csv"
OBSID_FORMAT = "%Y%m%dT%H%M%SZ"
VDIF_FORMAT = "drs4-{0}-chassis{1}-in{2}.vdif"
ZARR_CHUNKS = {"time": 600, "chan": CHAN_TOTAL}
ZARR_ENCODING = {
    "time": {
        "dtype": "int64",
        "units": "nanoseconds since 2000-01-01",
    }
}
ZARR_FORMAT = "drs4-{0}-chassis{1}.zarr"
ZARR_SHARDS = {"time": 36000, "chan": CHAN_TOTAL}

# constants (environment variable)
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
