__all__ = ["cross", "crosses"]


# standard library
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from logging import getLogger
from os import getenv
from pathlib import Path
from threading import Barrier, BrokenBarrierError, Event
from time import sleep
from typing import Any
from warnings import filterwarnings

# dependencies
import xarray as xr
from tqdm import tqdm
from ..ctrl.self import run, set_gain
from ..specs.common import (
    CHAN_TOTAL,
    ENV_CTRL_ADDR,
    ENV_CTRL_USER,
    ZARR_CHUNKS,
    ZARR_ENCODING,
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

# global settings
CSV_AUTO = "~/DRS4/mrdsppy/output/new_pow.csv"
CSV_CROSS = "~/DRS4/mrdsppy/output/new_phase.csv"
CSV_ROW_TOTAL = CHAN_TOTAL + 1  # '1' indicates header row
LOGGER = getLogger(__name__)
filterwarnings("ignore", category=FutureWarning)


def cross(
    *,
    # for measurement (required)
    chassis: Chassis,
    duration: Event | int,
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
    progress: bool | int = False,
    workdir: StrPath | None = None,
    zarr: StrPath | None = None,
    # for DRS4 settings (optional)
    ctrl_addr: str | None = None,
    ctrl_user: str | None = None,
    dsp_mode: DSPMode = "IQ",
    gain: xr.DataTree | StrPath | None = None,
    settings: bool = True,
    timeout: float | None = None,
    # for external synchronization (optional)
    sync: Barrier | None = None,
) -> Path:
    """"""
    if ctrl_addr is None:
        ctrl_addr = getenv(ENV_CTRL_ADDR.format(chassis), "")

    if ctrl_user is None:
        ctrl_user = getenv(ENV_CTRL_USER.format(chassis), "")

    if zarr is None:
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        zarr = f"drs4-{now}.zarr"

    LOGGER.debug("(")

    for key, val in locals().items():
        LOGGER.debug(f"  {key}: {val!r}")

    LOGGER.debug(")")

    if append and overwrite:
        raise ValueError("Append and overwrite cannot be enabled at once.")

    if (zarr := Path(zarr)).exists() and not append and not overwrite:
        raise FileExistsError(zarr)

    group_if1 = Path(f"chassis{chassis}") / "if1"
    group_if2 = Path(f"chassis{chassis}") / "if2"

    if isinstance(gain, xr.DataTree):
        gain_if1 = gain[("/" / group_if1).as_posix()].to_dataset()
        gain_if2 = gain[("/" / group_if2).as_posix()].to_dataset()
    elif is_strpath(gain):
        gain_if1 = Path(gain) / group_if1
        gain_if2 = Path(gain) / group_if2
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

    with set_workdir(workdir) as workdir:
        with (
            open(
                csv_auto_if1 := workdir / f"{zarr.stem}-auto-chassis{chassis}-if1.csv",
                mode="w",
            ) as f_auto_if1,
            open(
                csv_cross_if1 := workdir
                / f"{zarr.stem}-cross-chassis{chassis}-if1.csv",
                mode="w",
            ) as f_cross_if1,
            open(
                csv_auto_if2 := workdir / f"{zarr.stem}-auto-chassis{chassis}-if2.csv",
                mode="w",
            ) as f_auto_if2,
            open(
                csv_cross_if2 := workdir
                / f"{zarr.stem}-cross-chassis{chassis}-if2.csv",
                mode="w",
            ) as f_cross_if2,
        ):
            if sync is not None:
                try:
                    sync.wait(timeout)
                except BrokenBarrierError:
                    return zarr.resolve()

            try:
                with tqdm(
                    desc=f"Chassis {chassis}",
                    disable=not progress,
                    leave=True,
                    position=max(int(progress) - 1, 0),
                    total=None if isinstance(duration, Event) else int(duration),
                    unit="s",
                ) as bar:
                    cycle = 0
                    while True:
                        if isinstance(duration, Event):
                            if duration.is_set():
                                LOGGER.debug("Data acquisition finished by event.")
                                break
                        else:
                            if cycle >= duration:
                                LOGGER.debug("Data acquisition finished by duration.")
                                break

                        time = datetime.now(timezone.utc).strftime(TIME_FORMAT)
                        result = run(
                            # for interface 1
                            f"./get_corr_rslt.py --In 1",
                            "sleep 1",
                            f"cat {CSV_AUTO}",
                            f"cat {CSV_CROSS}",
                            # for interface 2
                            f"./get_corr_rslt.py --In 3",
                            "sleep 1",
                            f"cat {CSV_AUTO}",
                            f"cat {CSV_CROSS}",
                            chassis=chassis,
                            timeout=timeout,
                        )
                        result.check_returncode()
                        rows = result.stdout.split()

                        # write header
                        if cycle == 0:
                            f_auto_if1.write(f"time,{rows[CSV_ROW_TOTAL * 0 + 1]}\n")
                            f_cross_if1.write(f"time,{rows[CSV_ROW_TOTAL * 1 + 1]}\n")
                            f_auto_if2.write(f"time,{rows[CSV_ROW_TOTAL * 2 + 2]}\n")
                            f_cross_if2.write(f"time,{rows[CSV_ROW_TOTAL * 3 + 2]}\n")

                        # write data
                        for ch in range(CHAN_TOTAL):
                            f_auto_if1.write(
                                f"{time},{rows[(CSV_ROW_TOTAL * 0 + 1) + ch + 1]}\n"
                            )
                            f_cross_if1.write(
                                f"{time},{rows[(CSV_ROW_TOTAL * 1 + 1) + ch + 1]}\n"
                            )
                            f_auto_if2.write(
                                f"{time},{rows[(CSV_ROW_TOTAL * 2 + 2) + ch + 1]}\n"
                            )
                            f_cross_if2.write(
                                f"{time},{rows[(CSV_ROW_TOTAL * 3 + 2) + ch + 1]}\n"
                            )

                        bar.update(1)
                        cycle += 1
            except KeyboardInterrupt:
                LOGGER.warning("Data acquisition interrupted by user.")

        ds_if1, ds_if2 = xr.align(
            open_csvs(
                csv_auto_if1,
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
                csv_auto_if2,
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

        if (zarr / group_if1).exists() and append:
            ds_if1.chunk(ZARR_CHUNKS).to_zarr(
                zarr,
                group=("/" / group_if1).as_posix(),
                mode="a",
                append_dim="time",
                consolidated=False,
                safe_chunks=False,
            )
        else:
            ds_if1.chunk(ZARR_CHUNKS).to_zarr(
                zarr,
                group=("/" / group_if1).as_posix(),
                mode="w",
                encoding=encoding_if1,
                consolidated=False,
                safe_chunks=False,
            )

        if (zarr / group_if2).exists() and append:
            ds_if2.chunk(ZARR_CHUNKS).to_zarr(
                zarr,
                group=("/" / group_if2).as_posix(),
                mode="a",
                append_dim="time",
                consolidated=False,
                safe_chunks=False,
            )
        else:
            ds_if2.chunk(ZARR_CHUNKS).to_zarr(
                zarr,
                group=("/" / group_if2).as_posix(),
                mode="a",
                encoding=encoding_if2,
                consolidated=False,
                safe_chunks=False,
            )

        return zarr.resolve()


def crosses(
    *,
    # for measurement (required)
    chasses: Sequence[Chassis],
    duration: Event | int,
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
    timeout: float | None = None,
) -> Path:
    """"""
    if zarr is None:
        now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        zarr = f"drs4-{now}.zarr"

    LOGGER.debug("(")

    for key, val in locals().items():
        LOGGER.debug(f"  {key}: {val!r}")

    LOGGER.debug(")")

    if append and overwrite:
        raise ValueError("Append and overwrite cannot be enabled at once.")

    if (zarr := Path(zarr)).exists() and not append and not overwrite:
        raise FileExistsError(zarr)

    interrupt = duration if isinstance(duration, Event) else Event()
    chasses = sorted(set(chasses))
    sync = Barrier(len(chasses) + 1)

    with ThreadPoolExecutor(max_workers=len(chasses)) as executor:
        futures: list[Future[Path]] = []

        for n, chassis in enumerate(chasses):
            future = executor.submit(
                cross,
                # for measurement (required)
                chassis=chassis,
                duration=interrupt,
                # for measurement (optional)
                freq_range_if1=freq_range_if1,
                freq_range_if2=freq_range_if2,
                integ_time=integ_time,
                signal_if=signal_if,
                signal_sb=signal_sb,
                signal_chan=signal_chan,
                # for file saving (optional)
                append=append,
                integrate=integrate,
                join=join,
                overwrite=overwrite,
                progress=n + 1 if progress else False,
                workdir=workdir,
                zarr=zarr,
                # for DRS4 settings (optional)
                dsp_mode=dsp_mode,
                gain=gain,
                settings=settings,
                timeout=timeout,
                # for external synchronization
                sync=sync,
            )
            futures.append(future)

        try:
            sync.wait(timeout=timeout)

            if isinstance(duration, int):
                interrupt.wait(duration)
                interrupt.set()
            else:
                interrupt.wait()

        except BrokenBarrierError:
            interrupt.set()
        except KeyboardInterrupt:
            interrupt.set()
            sync.abort()

        for future in futures:
            future.result()

    return zarr.resolve()
