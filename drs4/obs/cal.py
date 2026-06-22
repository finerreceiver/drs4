__all__ = ["yfactor"]

# standard library
from pathlib import Path

# dependencies
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from ..daq import auto
from ..specs.common import Chassis
from ..utils import StrPath

# constants
COLOR_HOT = "#b7282e"  # https://www.colordic.org/colorsample/2009
COLOR_COLD = "#1e50a2"  # https://www.colordic.org/colorsample/2069
COLOR_OTHER = "#4f455c"  # https://www.colordic.org/colorsample/2295
DRS4_BINARY_POINT = 15
DRS4_INTNUM = 2000000
DRS4_NFFT = 1024


def yfactor(
    *,
    chassis: Chassis,
    duration: int = 10,
    figsize: tuple[float, float] = (12, 6),
    zarr_hot: StrPath | None = None,
    zarr_cold: StrPath | None = None,
) -> Path:
    """Measure and plot the Y factor of a chassis of DRS4."""
    if zarr_hot is None or zarr_cold is None:
        input("Press any key to start the hot measurement.")
        zarr_hot = auto(chassis=chassis, duration=duration)
        input("Press any key to start the cold measurement.")
        zarr_cold = auto(chassis=chassis, duration=duration)

        return yfactor(
            chassis=chassis,
            duration=duration,
            zarr_hot=zarr_hot,
            zarr_cold=zarr_cold,
        )

    if1_hot = xr.open_dataset(
        Path(zarr_hot) / "if1",
        engine="zarr",
        backend_kwargs={"consolidated": False},
    )
    if2_hot = xr.open_dataset(
        Path(zarr_hot) / "if2",
        engine="zarr",
        backend_kwargs={"consolidated": False},
    )
    if1_cold = xr.open_dataset(
        Path(zarr_cold) / "if1",
        engine="zarr",
        backend_kwargs={"consolidated": False},
    )
    if2_cold = xr.open_dataset(
        Path(zarr_cold) / "if2",
        engine="zarr",
        backend_kwargs={"consolidated": False},
    )

    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True)

    ax = axes[0, 0]
    (
        to_dB(if1_hot["auto_lsb"].mean("time"), if1_hot.integ_time)
        .swap_dims(chan="freq")
        .plot.step(ax=ax, color=COLOR_HOT, label="Hot")
    )
    (
        to_dB(if1_cold["auto_lsb"].mean("time"), if1_cold.integ_time)
        .swap_dims(chan="freq")
        .plot.step(ax=ax, color=COLOR_COLD, label="Cold")
    )
    (
        to_dB(if2_hot["auto_lsb"].mean("time"), if2_hot.integ_time)
        .swap_dims(chan="freq")
        .plot.step(ax=ax, color=COLOR_HOT)
    )
    (
        to_dB(if2_cold["auto_lsb"].mean("time"), if2_cold.integ_time)
        .swap_dims(chan="freq")
        .plot.step(ax=ax, color=COLOR_COLD)
    )
    ax.set_title(f"Chassis {chassis}, LSB")
    ax.set_ylabel("Power [dB]")

    ax = axes[0, 1]
    (
        to_dB(if1_hot["auto_usb"].mean("time"), if1_hot.integ_time)
        .swap_dims(chan="freq")
        .plot.step(ax=ax, color=COLOR_HOT, label="Hot")
    )
    (
        to_dB(if1_cold["auto_usb"].mean("time"), if1_cold.integ_time)
        .swap_dims(chan="freq")
        .plot.step(ax=ax, color=COLOR_COLD, label="Cold")
    )
    (
        to_dB(if2_hot["auto_usb"].mean("time"), if2_hot.integ_time)
        .swap_dims(chan="freq")
        .plot.step(ax=ax, color=COLOR_HOT)
    )
    (
        to_dB(if2_cold["auto_usb"].mean("time"), if2_cold.integ_time)
        .swap_dims(chan="freq")
        .plot.step(ax=ax, color=COLOR_COLD)
    )
    ax.set_title(f"Chassis {chassis}, USB")
    ax.set_ylabel(None)

    for ax in axes[0]:
        ax.grid(True)
        ax.legend()
        ax.margins(x=0)
        ax.set_ylim(-60, -30)

    ax = axes[1, 0]
    (
        (
            to_dB(if1_hot["auto_lsb"].mean("time"), if1_hot.integ_time)
            - to_dB(if1_cold["auto_lsb"].mean("time"), if1_cold.integ_time)
        )
        .swap_dims(chan="freq")
        .plot.step(ax=ax, color=COLOR_OTHER)
    )
    (
        (
            to_dB(if2_hot["auto_lsb"].mean("time"), if2_hot.integ_time)
            - to_dB(if2_cold["auto_lsb"].mean("time"), if2_cold.integ_time)
        )
        .swap_dims(chan="freq")
        .plot.step(ax=ax, color=COLOR_OTHER)
    )
    ax.set_ylabel("Y factor [dB]")

    ax = axes[1, 1]
    (
        (
            to_dB(if1_hot["auto_usb"].mean("time"), if1_hot.integ_time)
            - to_dB(if1_cold["auto_usb"].mean("time"), if1_cold.integ_time)
        )
        .swap_dims(chan="freq")
        .plot.step(ax=ax, color=COLOR_OTHER)
    )
    (
        (
            to_dB(if2_hot["auto_usb"].mean("time"), if2_hot.integ_time)
            - to_dB(if2_cold["auto_usb"].mean("time"), if2_cold.integ_time)
        )
        .swap_dims(chan="freq")
        .plot.step(ax=ax, color=COLOR_OTHER)
    )
    ax.set_ylabel(None)

    for ax in axes[1]:
        ax.grid(True)
        ax.margins(x=0)
        ax.set_ylim(-1, 5)

    fig.tight_layout()
    fig.savefig(path := Path(f"drs4-yfactor-chassis{chassis}.pdf"))
    return path.resolve()


def to_dB(da: xr.DataArray, integ_time: int, /) -> xr.DataArray:
    """Convert power scale to dB scale."""
    return 10 * np.log10(
        (da + 2**-40)
        * 2**DRS4_BINARY_POINT
        / DRS4_INTNUM
        / DRS4_NFFT**2
        / int(integ_time / 100)
    )
