__all__ = ["GAIN_ONES", "GAIN_ZEROS", "Gain", "open_gain"]


# standard library
from dataclasses import dataclass, field
from typing import Literal as L, Union, overload

# dependencies
import numpy as np
import pandas as pd
import xarray as xr
from xarray_dataclasses import AsDataset, Attr, Coordof, Data, Dataof
from .common import CHAN_TOTAL, Chan
from ..utils import StrPath


@dataclass
class GainUSB:
    data: Data[L["chan"], np.complex128]
    long_name: Attr[str] = "Complex gain of USB"
    units: Attr[str] = "Arbitrary unit"


@dataclass
class GainLSB:
    data: Data[L["chan"], np.complex128]
    long_name: Attr[str] = "Complex gain of LSB"
    units: Attr[str] = "Arbitrary unit"


@dataclass
class Gain(AsDataset):
    """Complex gains for digital sideband separation."""

    # dims
    chan: Coordof[Chan]
    """Channel number."""

    # vars
    usb: Dataof[GainUSB]
    """Complex gain of USB."""

    lsb: Dataof[GainLSB]
    """Complex gain of LSB."""

    # attrs
    spec_version: Attr[int] = field(default=0, init=False)
    """Version of the data specification."""


@overload
def open_gain(
    ms: StrPath,
    /,
    *,
    format: L["Dataset"],
) -> xr.Dataset: ...


@overload
def open_gain(
    ms: StrPath,
    /,
    *,
    format: L["DataFrame"],
) -> pd.DataFrame: ...


@overload
def open_gain(
    ms: StrPath,
    /,
) -> xr.Dataset: ...


def open_gain(
    ms: StrPath,
    /,
    *,
    format: L["DataFrame", "Dataset"] = "Dataset",
) -> Union[pd.DataFrame, xr.Dataset]:
    """Open gain file (DRS4 MS file) as a DataFrame or a Dataset.

    Args:
        ms: Path of input gain file (DRS4 MS file).
        format: Output data format (DataFrame|Dataset).

    Returns:
        DataFrame or Dataset of the input gain file.

    """
    if format not in ("DataFrame", "Dataset"):
        raise ValueError("Output data format must be DataFrame|Dataset.")

    ds = xr.open_zarr(ms)
    masked = ds.where(ds.chan == ds.signal_chan)
    masked_usb = masked.where(masked.signal_sb == "USB", drop=True)
    masked_lsb = masked.where(masked.signal_sb == "LSB", drop=True)
    gain_usb = -(masked_usb.cross_2sb / masked_usb.auto_usb).conj()
    gain_lsb = -(masked_lsb.cross_2sb / masked_lsb.auto_lsb)

    gain = Gain.new(
        chan=ds.chan.data,
        usb=gain_usb.mean("time").fillna(0).data,
        lsb=gain_lsb.mean("time").fillna(0).data,
    )

    if format == "Dataset":
        return gain
    else:
        return to_dataframe(gain)


def to_dataframe(gain: xr.Dataset, /) -> pd.DataFrame:
    """Convert a gain Dataset to DataFrame for a coefficient table."""
    coef_re0 = (gain.usb.real * 8192).astype(int) & 0xFFFFFFFF
    coef_im0 = (gain.usb.imag * 8192).astype(int) & 0xFFFFFFFF
    coef_re1 = (gain.lsb.real * 8192).astype(int) & 0xFFFFFFFF
    coef_im1 = (gain.lsb.imag * 8192).astype(int) & 0xFFFFFFFF

    return pd.DataFrame(
        data={
            "coef_re0": coef_re0.to_series().map("{:#010x}".format).values,
            "coef_im0": coef_im0.to_series().map("{:#010x}".format).values,
            "coef_re1": coef_re1.to_series().map("{:#010x}".format).values,
            "coef_im1": coef_im1.to_series().map("{:#010x}".format).values,
        }
    )


GAIN_ONES = Gain.new(
    chan=np.arange(CHAN_TOTAL),
    usb=np.ones(CHAN_TOTAL),
    lsb=np.ones(CHAN_TOTAL),
)
GAIN_ZEROS = Gain.new(
    chan=np.arange(CHAN_TOTAL),
    usb=np.zeros(CHAN_TOTAL),
    lsb=np.zeros(CHAN_TOTAL),
)
