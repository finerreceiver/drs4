__all__ = ["cross"]


# standard library
from datetime import datetime, timezone
from logging import getLogger
from os import getenv
from pathlib import Path
from typing import Optional, get_args


# dependencies
import xarray as xr
from tqdm import tqdm
from ..ctrl.self import ENV_CTRL_ADDR, ENV_CTRL_USER, run
from ..specs.csv import TIME_FORMAT
from ..specs.zarr import (
    CHAN_TOTAL,
    Channel,
    Chassis,
    FreqRange,
    IntegTime,
    Interface,
    SideBand,
    open_csvs,
)
from ..utils import StrPath, XarrayJoin, set_workdir, unique


# constants
CSV_AUTOS = "~/DRS4/mrdsppy/output/new_pow.csv"
CSV_AUTOS_FORMAT = "drs4-{0}-chassis{1}-autos-if{2}.csv"
CSV_CROSS = "~/DRS4/mrdsppy/output/new_phase.csv"
CSV_CROSS_FORMAT = "drs4-{0}-chassis{1}-cross-if{2}.csv"
CSV_ROW_TOTAL = CHAN_TOTAL + 1
LOGGER = getLogger(__name__)
OBSID_FORMAT = "%Y%m%dT%H%M%SZ"
ZARR_FORMAT = "drs4-{0}-chassis{1}-if{2}.zarr.zip"


def cross(
    *,
    # for measurement (required)
    chassis: Chassis,
    duration: int,
    # for measurement (optional)
    freq_range_if1: FreqRange = "inner",
    freq_range_if2: FreqRange = "outer",
    integ_time: IntegTime = 100,
    signal_if: Optional[Interface] = None,
    signal_sb: Optional[SideBand] = None,
    signal_chan: Optional[Channel] = None,
    # for file saving (optional)
    append: bool = False,
    integrate: bool = False,
    join: XarrayJoin = "inner",
    overwrite: bool = False,
    progress: bool = False,
    workdir: Optional[StrPath] = None,
    zarr_if1: Optional[StrPath] = None,
    zarr_if2: Optional[StrPath] = None,
    # for connection (optional)
    ctrl_addr: Optional[str] = None,
    ctrl_user: Optional[str] = None,
    timeout: Optional[float] = None,
) -> tuple[Path, Path]:
    """"""
    obsid = datetime.now(timezone.utc).strftime(OBSID_FORMAT)

    if ctrl_addr is None:
        ctrl_addr = getenv(ENV_CTRL_ADDR.format(chassis), "")

    if ctrl_user is None:
        ctrl_user = getenv(ENV_CTRL_USER.format(chassis), "")

    if zarr_if1 is None:
        zarr_if1 = ZARR_FORMAT.format(obsid, chassis, 1)

    if zarr_if2 is None:
        zarr_if2 = ZARR_FORMAT.format(obsid, chassis, 2)

    if append and overwrite:
        raise ValueError("Append and overwrite cannot be enabled at once.")

    if chassis not in get_args(Chassis):
        raise ValueError("Chassis number must be 1|2.")

    if freq_range_if1 not in get_args(FreqRange):
        raise ValueError("Frequency range must be inner|outer.")

    if freq_range_if2 not in get_args(FreqRange):
        raise ValueError("Frequency range must be inner|outer.")

    if integ_time not in get_args(IntegTime):
        raise ValueError("Spectral integration time must be 100|200|500|1000.")

    if (zarr_if1 := Path(zarr_if1)).exists() and not append and not overwrite:
        raise FileExistsError(zarr_if1)

    if (zarr_if2 := Path(zarr_if2)).exists() and not append and not overwrite:
        raise FileExistsError(zarr_if2)

    result = run(
        # for interface 1
        f"./set_intg_time.py --In 1 --It {integ_time // 100}",
        f"./get_intg_time.py --In 1",
        # for interface 2
        f"./set_intg_time.py --In 3 --It {integ_time // 100}",
        f"./get_intg_time.py --In 3",
        chassis=chassis,
        timeout=timeout,
    )
    result.check_returncode()

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
        for cycle in range(duration):
            time = datetime.now(timezone.utc).strftime(TIME_FORMAT)
            result = run(
                # for interface 1
                f"./get_corr_rslt.py --In 1",
                f"cat {CSV_AUTOS}",
                f"cat {CSV_CROSS}",
                # for interface 2
                f"./get_corr_rslt.py --In 3",
                f"cat {CSV_AUTOS}",
                f"cat {CSV_CROSS}",
                chassis=chassis,
                timeout=timeout,
            )
            result.check_returncode()
            rows = result.stdout.split()

            # write header
            if cycle == 1:
                f_autos_if1.write(f"time,{rows[CSV_ROW_TOTAL * 0]}\n")
                f_cross_if1.write(f"time,{rows[CSV_ROW_TOTAL * 1]}\n")
                f_autos_if2.write(f"time,{rows[CSV_ROW_TOTAL * 2]}\n")
                f_cross_if2.write(f"time,{rows[CSV_ROW_TOTAL * 3]}\n")

            # write data
            for ch in range(CHAN_TOTAL):
                f_autos_if1.write(f"{time},{rows[CSV_ROW_TOTAL * 0 + 1 + ch]}\n")
                f_cross_if1.write(f"{time},{rows[CSV_ROW_TOTAL * 1 + 1 + ch]}\n")
                f_autos_if2.write(f"{time},{rows[CSV_ROW_TOTAL * 2 + 1 + ch]}\n")
                f_cross_if2.write(f"{time},{rows[CSV_ROW_TOTAL * 3 + 1 + ch]}\n")

            bar.update(1)

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

        if zarr_if1.exists() and append:
            ds_if1.to_zarr(zarr_if1, mode="a", append_dim="time")
        else:
            ds_if1.to_zarr(zarr_if1, mode="w")

        if zarr_if2.exists() and append:
            ds_if2.to_zarr(zarr_if2, mode="a", append_dim="time")
        else:
            ds_if2.to_zarr(zarr_if2, mode="w")

        return zarr_if1.resolve(), zarr_if2.resolve()
