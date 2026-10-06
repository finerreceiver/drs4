__all__ = ["to_zarr"]


# standard library
from pathlib import Path
from typing import Any

# dependencies
import xarray as xr
from ..specs.common import ZARR_CHUNKS, ZARR_ENCODING, ZARR_SHARDS
from ..utils import StrPath, unique


def to_zarr(
    ds_if1: xr.Dataset,
    ds_if2: xr.Dataset,
    zarr: StrPath,
    /,
    *,
    append: bool = False,
    integrate: bool = False,
) -> Path:
    """Save or append datasets to a Zarr store."""
    for ds in (ds_if1, ds_if2):
        group = Path(f"chassis{ds.chassis}") / f"if{ds.interface}"
        encoding: dict[Any, Any] = ZARR_ENCODING.copy()

        if integrate:
            ds = ds.coarsen(
                dim={"time": ds.sizes["time"]},
                coord_func={"signal_chan": unique, "signal_sb": unique},
            ).mean()  # type: ignore

        for name, var in ds.variables.items():
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
            ds.chunk(ZARR_SHARDS).to_zarr(
                zarr,
                group=("/" / group).as_posix(),
                mode="a",
                append_dim="time",
                consolidated=False,
                safe_chunks=False,
            )
        else:
            ds.chunk(ZARR_SHARDS).to_zarr(
                zarr,
                group=("/" / group).as_posix(),
                mode="w" if ds.interface == 1 else "a",
                encoding=encoding,
                consolidated=False,
                safe_chunks=True,
            )

    return zarr.resolve()
