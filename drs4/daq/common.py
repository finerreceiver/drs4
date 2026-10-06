__all__ = ["to_zarr"]


# standard library
from pathlib import Path
from typing import Any

# dependencies
import xarray as xr
from ..specs.common import ZARR_CHUNKS, ZARR_ENCODING, ZARR_SHARDS
from ..utils import StrPath, unique


def to_zarr(
    ms_if1: xr.Dataset,
    ms_if2: xr.Dataset,
    zarr: StrPath,
    /,
    *,
    append: bool = False,
    integrate: bool = False,
) -> Path:
    """Save or append DRS4 measurement sets to a Zarr store."""
    for ms in (ms_if1, ms_if2):
        group = Path(f"chassis{ms.chassis}") / f"if{ms.interface}"
        encoding: dict[Any, Any] = ZARR_ENCODING.copy()

        if integrate:
            ms = ms.coarsen(
                dim={"time": ms.sizes["time"]},
                coord_func={"signal_chan": unique, "signal_sb": unique},
            ).mean()  # type: ignore

        for name, var in ms.variables.items():
            encoding.setdefault(name, {})
            encoding[name]["chunks"] = tuple(
                # fmt: off
                ZARR_CHUNKS.get(dim, var.sizes[dim])  # type: ignore
                for dim in var.dims
                # fmt: on
            )
            encoding[name]["shards"] = tuple(
                # fmt: off
                ZARR_SHARDS.get(dim, var.sizes[dim])  # type: ignore
                for dim in var.dims
                # fmt: on
            )

        if ((zarr := Path(zarr)) / group).exists() and append:
            ms.chunk(ZARR_SHARDS).to_zarr(
                zarr,
                group=("/" / group).as_posix(),
                mode="a",
                append_dim="time",
                consolidated=False,
                safe_chunks=False,
            )
        else:
            ms.chunk(ZARR_SHARDS).to_zarr(
                zarr,
                group=("/" / group).as_posix(),
                mode="w" if ms.interface == 1 else "a",
                encoding=encoding,
                consolidated=False,
                safe_chunks=True,
            )

    return zarr.resolve()
