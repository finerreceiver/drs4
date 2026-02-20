__all__ = ["cross"]


# standard library
from datetime import datetime, timezone
from logging import getLogger
from os import getenv
from pathlib import Path
from typing import Optional, Union, get_args

# dependencies
import xarray as xr
from tqdm import tqdm
from ..cal.dsbs import set_gain
from ..ctrl.self import run
from ..specs.common import (
    CHAN_TOTAL,
    CSV_AUTOS_FORMAT,
    CSV_CROSS_FORMAT,
    ENV_CTRL_ADDR,
    ENV_CTRL_USER,
    OBSID_FORMAT,
    TIME_UNITS,
    ZARR_FORMAT,
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
from ..utils import StrPath, XarrayJoin, set_workdir, unique

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
    # for DRS4 settings (optional)
    dsp_mode: DSPMode = "IQ",
    gain_if1: Optional[Union[xr.Dataset, StrPath]] = None,
    gain_if2: Optional[Union[xr.Dataset, StrPath]] = None,
    settings: bool = True,
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

    LOGGER.info("(")

    for key, val in locals().items():
        LOGGER.info(f"  {key}: {val!r}")

    LOGGER.info(")")

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

        if zarr_if1.exists() and append:
            ds_if1.to_zarr(
                zarr_if1,  # type: ignore
                mode="a",
                append_dim="time",
            )
        else:
            ds_if1.to_zarr(
                zarr_if1,  # type: ignore
                mode="w",
                encoding={"time": {"units": TIME_UNITS}},
            )

        if zarr_if2.exists() and append:
            ds_if2.to_zarr(
                zarr_if2,  # type: ignore
                mode="a",
                append_dim="time",
            )
        else:
            ds_if2.to_zarr(
                zarr_if2,  # type: ignore
                mode="w",
                encoding={"time": {"units": TIME_UNITS}},
            )

        return zarr_if1.resolve(), zarr_if2.resolve()
