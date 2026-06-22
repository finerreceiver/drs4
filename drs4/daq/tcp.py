__all__ = ["cross"]


# standard library
from datetime import datetime, timezone
from logging import getLogger
from os import getenv
from pathlib import Path
from time import sleep
from typing import Any
from warnings import catch_warnings, simplefilter

# dependencies
import xarray as xr
from tqdm import tqdm
from ..ctrl.self import run, set_gain
from ..specs.common import (
    CHAN_TOTAL,
    CSV_AUTOS_FORMAT,
    CSV_CROSS_FORMAT,
    ENV_CTRL_ADDR,
    ENV_CTRL_USER,
    OBSID_FORMAT,
    ZARR_CHUNKS,
    ZARR_ENCODING,
    ZARR_FORMAT,
    ZARR_SHARDS,
    Channel,
    Chassis,
    DSPMode,
    FreqRange,
    IntegTime,
    Interface,
    SideBand,
)
from ..specs.ms import open_csvs
from ..specs.csv import TIME_FORMAT
from ..utils import StrPath, XarrayJoin, is_strpath, set_workdir, unique

# constants
CSV_AUTOS = "~/DRS4/mrdsppy/output/new_pow.csv"
CSV_CROSS = "~/DRS4/mrdsppy/output/new_phase.csv"
CSV_ROW_TOTAL = CHAN_TOTAL + 1  # 1 means header
LOGGER = getLogger(__name__)


def cross(
    *,
    # for measurement (required)
    chassis: Chassis,
    duration: int,
    # for measurement (optional)
    freq_range_if1: FreqRange = "inner",
    freq_range_if2: FreqRange = "outer",
    integ_time: IntegTime = 100,
    signal_if: Interface | None = None,
    signal_sb: SideBand | None = None,
    signal_chan: Channel | None = None,
    # for file saving (optional)
    append: bool = False,
    integrate: bool = False,
    join: XarrayJoin = "inner",
    overwrite: bool = False,
    progress: bool = False,
    workdir: StrPath | None = None,
    zarr: StrPath | None = None,
    # for DRS4 settings (optional)
    dsp_mode: DSPMode = "IQ",
    gain: xr.DataTree | StrPath | None = None,
    settings: bool = True,
    # for connection (optional)
    ctrl_addr: str | None = None,
    ctrl_user: str | None = None,
    timeout: float | None = None,
) -> Path:
    """"""
    obsid = datetime.now(timezone.utc).strftime(OBSID_FORMAT)

    if ctrl_addr is None:
        ctrl_addr = getenv(ENV_CTRL_ADDR.format(chassis), "")

    if ctrl_user is None:
        ctrl_user = getenv(ENV_CTRL_USER.format(chassis), "")

    if zarr is None:
        zarr = ZARR_FORMAT.format(obsid, chassis)

    LOGGER.debug("(")

    for key, val in locals().items():
        LOGGER.debug(f"  {key}: {val!r}")

    LOGGER.debug(")")

    if append and overwrite:
        raise ValueError("Append and overwrite cannot be enabled at once.")

    if chassis not in (1, 2):
        raise ValueError("Chassis number must be 1|2.")

    if freq_range_if1 not in ("inner", "outer"):
        raise ValueError("Frequency range must be inner|outer.")

    if freq_range_if2 not in ("inner", "outer"):
        raise ValueError("Frequency range must be inner|outer.")

    if integ_time not in (100, 200, 500, 1000):
        raise ValueError("Spectral integration time must be 100|200|500|1000.")

    if (zarr := Path(zarr)).exists() and not append and not overwrite:
        raise FileExistsError(zarr)

    if isinstance(gain, xr.DataTree):
        gain_if1 = gain["/if1"].to_dataset()
        gain_if2 = gain["/if2"].to_dataset()
    elif is_strpath(gain):
        gain_if1 = Path(gain) / "if1"
        gain_if2 = Path(gain) / "if2"
    elif gain is None:
        gain_if1 = None
        gain_if2 = None
    else:
        raise TypeError("Gain must be either DataTree or Zarr path.")

    if settings:
        set_gain(
            gain_if1,
            chassis=chassis,
            interface=1,
            ones=True if gain_if1 is None else False,
            ctrl_addr=ctrl_addr,
            ctrl_user=ctrl_user,
            timeout=timeout,
        )
        set_gain(
            gain_if2,
            chassis=chassis,
            interface=2,
            ones=True if gain_if2 is None else False,
            ctrl_addr=ctrl_addr,
            ctrl_user=ctrl_user,
            timeout=timeout,
        )
        result = run(
            # for interface 1
            f"./set_intg_time.py --In 1 --It {integ_time // 100}",
            f"./set_mode.py --In 1 -m {dsp_mode}",
            # for interface 2
            f"./set_intg_time.py --In 3 --It {integ_time // 100}",
            f"./set_mode.py --In 3 -m {dsp_mode}",
            chassis=chassis,
            timeout=timeout,
        )
        result.check_returncode()
        sleep(1.5)

    with (
        set_workdir(workdir) as workdir,
        tqdm(disable=not progress, total=int(duration), unit="s") as bar,
        open(
            csv_autos_if1 := workdir / CSV_AUTOS_FORMAT.format(obsid, chassis, 1),
            mode="w",
        ) as f_autos_if1,
        open(
            csv_cross_if1 := workdir / CSV_CROSS_FORMAT.format(obsid, chassis, 1),
            mode="w",
        ) as f_cross_if1,
        open(
            csv_autos_if2 := workdir / CSV_AUTOS_FORMAT.format(obsid, chassis, 2),
            mode="w",
        ) as f_autos_if2,
        open(
            csv_cross_if2 := workdir / CSV_CROSS_FORMAT.format(obsid, chassis, 2),
            mode="w",
        ) as f_cross_if2,
    ):
        try:
            for cycle in range(duration):
                time = datetime.now(timezone.utc).strftime(TIME_FORMAT)
                result = run(
                    # for interface 1
                    f"./get_corr_rslt.py --In 1",
                    "sleep 1",
                    f"cat {CSV_AUTOS}",
                    f"cat {CSV_CROSS}",
                    # for interface 2
                    f"./get_corr_rslt.py --In 3",
                    "sleep 1",
                    f"cat {CSV_AUTOS}",
                    f"cat {CSV_CROSS}",
                    chassis=chassis,
                    timeout=timeout,
                )
                result.check_returncode()
                rows = result.stdout.split()

                # write header
                if cycle == 0:
                    f_autos_if1.write(f"time,{rows[CSV_ROW_TOTAL * 0 + 1]}\n")
                    f_cross_if1.write(f"time,{rows[CSV_ROW_TOTAL * 1 + 1]}\n")
                    f_autos_if2.write(f"time,{rows[CSV_ROW_TOTAL * 2 + 2]}\n")
                    f_cross_if2.write(f"time,{rows[CSV_ROW_TOTAL * 3 + 2]}\n")

                # write data
                for ch in range(CHAN_TOTAL):
                    f_autos_if1.write(
                        f"{time},{rows[(CSV_ROW_TOTAL * 0 + 1) + ch + 1]}\n"
                    )
                    f_cross_if1.write(
                        f"{time},{rows[(CSV_ROW_TOTAL * 1 + 1) + ch + 1]}\n"
                    )
                    f_autos_if2.write(
                        f"{time},{rows[(CSV_ROW_TOTAL * 2 + 2) + ch + 1]}\n"
                    )
                    f_cross_if2.write(
                        f"{time},{rows[(CSV_ROW_TOTAL * 3 + 2) + ch + 1]}\n"
                    )

                bar.update(1)
        except KeyboardInterrupt:
            LOGGER.warning("Data acquisition interrupted by user.")
        finally:
            f_autos_if1.flush()
            f_cross_if1.flush()
            f_autos_if2.flush()
            f_cross_if2.flush()

        ds_if1, ds_if2 = xr.align(
            open_csvs(
                csv_autos_if1,
                csv_cross_if1,
                # for measurement (required)
                chassis=chassis,
                interface=1,
                freq_range=freq_range_if1,
                # for measurement (optional)
                integ_time=integ_time,
                signal_sb=signal_sb if signal_if == 1 else None,
                signal_chan=signal_chan if signal_if == 1 else None,
            ),
            open_csvs(
                csv_autos_if2,
                csv_cross_if2,
                # for measurement (required)
                chassis=chassis,
                interface=2,
                freq_range=freq_range_if2,
                # for measurement (optional)
                integ_time=integ_time,
                signal_sb=signal_sb if signal_if == 2 else None,
                signal_chan=signal_chan if signal_if == 2 else None,
            ),
            join=join,
        )

        if integrate:
            dim = {"time": ds_if1.sizes["time"]}
            coord_func = {"signal_chan": unique, "signal_sb": unique}
            ds_if1 = ds_if1.coarsen(dim, coord_func=coord_func).mean()  # type: ignore
            ds_if2 = ds_if2.coarsen(dim, coord_func=coord_func).mean()  # type: ignore

        encoding_if1: dict[Any, Any] = ZARR_ENCODING.copy()
        encoding_if2: dict[Any, Any] = ZARR_ENCODING.copy()

        for name, var in ds_if1.variables.items():
            encoding_if1.setdefault(name, {})
            encoding_if1[name]["chunks"] = tuple(
                # fmt: off
                ZARR_CHUNKS.get(dim, var.sizes[dim]) # type: ignore
                for dim in var.dims
                # fmt: on
            )
            encoding_if1[name]["shards"] = tuple(
                # fmt: off
                ZARR_SHARDS.get(dim, var.sizes[dim]) # type: ignore
                for dim in var.dims
                # fmt: on
            )

        for name, var in ds_if2.variables.items():
            encoding_if2.setdefault(name, {})
            encoding_if2[name]["chunks"] = tuple(
                # fmt: off
                ZARR_CHUNKS.get(dim, var.sizes[dim]) # type: ignore
                for dim in var.dims
                # fmt: on
            )
            encoding_if2[name]["shards"] = tuple(
                # fmt: off
                ZARR_SHARDS.get(dim, var.sizes[dim]) # type: ignore
                for dim in var.dims
                # fmt: on
            )
        with catch_warnings():
            simplefilter("ignore", category=FutureWarning)

            if zarr.exists() and append:
                ds_if1.chunk(ZARR_CHUNKS).to_zarr(
                    zarr,
                    group="/if1",
                    mode="a",
                    append_dim="time",
                    consolidated=False,
                )
                ds_if2.chunk(ZARR_CHUNKS).to_zarr(
                    zarr,
                    group="/if2",
                    mode="a",
                    append_dim="time",
                    consolidated=False,
                )
            else:
                ds_if1.chunk(ZARR_CHUNKS).to_zarr(
                    zarr,
                    group="/if1",
                    mode="w",
                    encoding=encoding_if1,
                    consolidated=False,
                )
                ds_if2.chunk(ZARR_CHUNKS).to_zarr(
                    zarr,
                    group="/if2",
                    mode="a",
                    encoding=encoding_if2,
                    consolidated=False,
                )

        return zarr.resolve()
