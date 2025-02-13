__all__ = ["auto"]


# standard library
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from logging import getLogger
from multiprocessing import Manager
from os import getenv
from pathlib import Path
from socket import (
    IP_ADD_MEMBERSHIP,
    IPPROTO_IP,
    SO_REUSEADDR,
    SOCK_DGRAM,
    SOL_SOCKET,
    inet_aton,
    socket,
)
from threading import Event
from time import sleep
from typing import Optional, Union, get_args


# dependencies
import xarray as xr
from tqdm import tqdm
from ..ctrl.self import run
from ..specs.common import (
    ENV_DEST_ADDR,
    ENV_DEST_PORT1,
    ENV_DEST_PORT2,
    ENV_DEST_PORT3,
    ENV_DEST_PORT4,
    OBSID_FORMAT,
    TIME_UNITS,
    VDIF_FORMAT,
    ZARR_FORMAT,
    Channel,
    Chassis,
    DSPMode,
    FreqRange,
    Interface,
    IntegTime,
    SideBand,
)
from ..specs.vdif import VDIF_FRAME_BYTES
from ..specs.zarr import open_vdifs
from ..utils import StrPath, XarrayJoin, set_workdir, unique


# constants
GROUP = "239.0.0.1"
LOGGER = getLogger(__name__)


def auto(
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
    dsp_mode: DSPMode = "SB",
    gain_if1: Optional[StrPath] = None,
    gain_if2: Optional[StrPath] = None,
    settings: bool = True,
    # for connection (optional)
    dest_addr: Optional[str] = None,
    dest_port1: Optional[int] = None,
    dest_port2: Optional[int] = None,
    dest_port3: Optional[int] = None,
    dest_port4: Optional[int] = None,
    timeout: Optional[float] = None,
) -> tuple[Path, Path]:
    """"""
    obsid = datetime.now(timezone.utc).strftime(OBSID_FORMAT)

    if dest_addr is None:
        dest_addr = getenv(ENV_DEST_ADDR.format(chassis), "")

    if dest_port1 is None:
        dest_port1 = int(getenv(ENV_DEST_PORT1.format(chassis), ""))

    if dest_port2 is None:
        dest_port2 = int(getenv(ENV_DEST_PORT2.format(chassis), ""))

    if dest_port3 is None:
        dest_port3 = int(getenv(ENV_DEST_PORT3.format(chassis), ""))

    if dest_port4 is None:
        dest_port4 = int(getenv(ENV_DEST_PORT4.format(chassis), ""))

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

    if settings:
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
        Manager() as manager,
        ProcessPoolExecutor(4) as executor,
        set_workdir(workdir) as workdir,
        tqdm(disable=not progress, total=int(duration), unit="s") as bar,
    ):
        cancel = manager.Event()
        executor.submit(
            dump,
            vdif_in1 := workdir / VDIF_FORMAT.format(obsid, chassis, 1),
            dest_addr=dest_addr,
            dest_port=dest_port1,
            cancel=cancel,
            timeout=timeout,
            overwrite=overwrite,
        )
        executor.submit(
            dump,
            vdif_in2 := workdir / VDIF_FORMAT.format(obsid, chassis, 2),
            dest_addr=dest_addr,
            dest_port=dest_port2,
            cancel=cancel,
            timeout=timeout,
            overwrite=overwrite,
        )
        executor.submit(
            dump,
            vdif_in3 := workdir / VDIF_FORMAT.format(obsid, chassis, 3),
            dest_addr=dest_addr,
            dest_port=dest_port3,
            cancel=cancel,
            timeout=timeout,
            overwrite=overwrite,
        )
        executor.submit(
            dump,
            vdif_in4 := workdir / VDIF_FORMAT.format(obsid, chassis, 4),
            dest_addr=dest_addr,
            dest_port=dest_port4,
            cancel=cancel,
            timeout=timeout,
            overwrite=overwrite,
        )

        try:
            for _ in range(int(duration)):
                sleep(1)
                bar.update(1)
        except KeyboardInterrupt:
            LOGGER.warning("Data acquisition interrupted by user.")
        finally:
            cancel.set()

        ds_if1, ds_if2 = xr.align(
            open_vdifs(
                vdif_in1,
                vdif_in2,
                # for measurement (required)
                chassis=chassis,
                interface=1,
                freq_range=freq_range_if1,
                # for measurement (optional)
                integ_time=integ_time,
                signal_sb=signal_sb if signal_if == 1 else None,
                signal_chan=signal_chan if signal_if == 1 else None,
                # for file loading (optional)
                join=join,
            ),
            open_vdifs(
                vdif_in3,
                vdif_in4,
                # for measurement (required)
                chassis=chassis,
                interface=2,
                freq_range=freq_range_if2,
                # for measurement (optional)
                integ_time=integ_time,
                signal_sb=signal_sb if signal_if == 2 else None,
                signal_chan=signal_chan if signal_if == 2 else None,
                # for file loading (optional)
                join=join,
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
            ds_if1.to_zarr(zarr_if1, mode="w", encoding={"time": {"units": TIME_UNITS}})

        if zarr_if2.exists() and append:
            ds_if2.to_zarr(zarr_if2, mode="a", append_dim="time")
        else:
            ds_if2.to_zarr(zarr_if2, mode="w", encoding={"time": {"units": TIME_UNITS}})

        return zarr_if1.resolve(), zarr_if2.resolve()


def dump(
    vdif: Union[Path, str],
    /,
    *,
    # for connection (required)
    dest_addr: str,
    dest_port: int,
    # for connection (optional)
    group: str = GROUP,
    # for file saving (optional)
    cancel: Optional[Event] = None,
    timeout: Optional[float] = None,
    progress: bool = False,
    overwrite: bool = False,
) -> None:
    """Receive and dump DRS4 data per input into a VDIF file.

    Args:
        vdif: Path of the output VDIF file.
        dest_addr: Destination IP address.
        dest_port: Destination port number.
        group: Multicast group IP address.
        cancel: Event object to cancel dumping.
        timeout: Timeout period in units of seconds.
        progress: Whether to show the progress bar on screen.
        overwrite: Whether to overwrite the existing VDIF file.

    Raises:
        FileExistsError: Raised if overwrite is not allowed
            and the output VDIF file already exists.
        TimeoutError: Raised if no DRS4 data (i.e. VDIF frame)
            is received for the timeout period.

    """
    if not overwrite and Path(vdif).exists():
        raise FileExistsError(vdif)

    prefix = f"[{dest_addr=}, {dest_port=}]"
    mreq = inet_aton(group) + inet_aton(dest_addr)

    with (
        open(vdif, "wb") as file,
        socket(type=SOCK_DGRAM) as sock,
        tqdm(desc=prefix, disable=not progress, unit="byte") as bar,
    ):
        # create socket
        sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        sock.bind(("", dest_port))
        sock.setsockopt(IPPROTO_IP, IP_ADD_MEMBERSHIP, mreq)
        sock.settimeout(timeout)

        # start dumping
        LOGGER.info(f"{prefix} Start dumping data.")

        while cancel is None or not cancel.is_set():
            frame, _ = sock.recvfrom(VDIF_FRAME_BYTES)

            if len(frame) == VDIF_FRAME_BYTES:
                file.write(frame)
                bar.update(VDIF_FRAME_BYTES)
            else:
                LOGGER.warning(f"{prefix} Truncated frame.")

        # finish dumping
        LOGGER.info(f"{prefix} Finish dumping data.")
