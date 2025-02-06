__all__ = ["run"]


# standard library
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from logging import getLogger
from multiprocessing import Manager
from os import PathLike, getenv
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
from tempfile import TemporaryDirectory
from threading import Event
from time import sleep
from typing import Optional, Union


# dependencies
import xarray as xr
from tqdm import tqdm
from .specs.vdif import VDIF_FRAME_BYTES
from .specs.zarr import Chassis, FreqRange, IntegTime, open_vdifs


# type hints
StrPath = Union[PathLike[str], str]


# constants
GROUP = "239.0.0.1"
LOGGER = getLogger(__name__)
OBSID_FORMAT = "%Y%m%dT%H%M%S"


def dump(
    vdif: Union[Path, str],
    addr: str,
    port: int,
    /,
    *,
    group: str = GROUP,
    cancel: Optional[Event] = None,
    timeout: Optional[float] = None,
    progress: bool = False,
    overwrite: bool = False,
) -> None:
    """Receive and dump DRS4 data per input into a VDIF file.

    Args:
        vdif: Path of the output VDIF file.
        addr: Destination IP address.
        port: Destination port number.
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

    prefix = f"[{addr=}, {port=}]"
    mreq = inet_aton(group) + inet_aton(addr)

    with (
        open(vdif, "wb") as file,
        socket(type=SOCK_DGRAM) as sock,
        tqdm(desc=prefix, disable=not progress, unit="byte") as bar,
    ):
        # create socket
        sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        sock.bind(("", port))
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


def run(
    *,
    # for file saving
    zarr_if1: Optional[StrPath] = None,
    zarr_if2: Optional[StrPath] = None,
    overwrite: bool = False,
    work_dir: Optional[StrPath] = None,
    # for measurement
    duration: int = 3600,
    chassis: Chassis = 1,
    integ_time: IntegTime = 100,
    freq_range_if1: FreqRange = "inner",
    freq_range_if2: FreqRange = "outer",
    # for connection
    drs4_dest_addr: Optional[str] = None,
    drs4_dest_port1: Optional[int] = None,
    drs4_dest_port2: Optional[int] = None,
    drs4_dest_port3: Optional[int] = None,
    drs4_dest_port4: Optional[int] = None,
) -> tuple[Path, Path]:
    obsid = datetime.now(timezone.utc).strftime(OBSID_FORMAT)

    if zarr_if1 is None:
        zarr_if1 = f"drs4-chassis{chassis}-if1-{obsid}.zarr.zip"

    if zarr_if2 is None:
        zarr_if2 = f"drs4-chassis{chassis}-if2-{obsid}.zarr.zip"

    if Path(zarr_if1).exists() and not overwrite:
        raise FileExistsError(zarr_if1)

    if Path(zarr_if2).exists() and not overwrite:
        raise FileExistsError(zarr_if2)

    drs4_dest_addr = drs4_dest_addr or getenv("DRS4_DEST_ADDR", "")
    drs4_dest_port1 = drs4_dest_port1 or int(getenv("DRS4_DEST_PORT1", 0))
    drs4_dest_port2 = drs4_dest_port2 or int(getenv("DRS4_DEST_PORT2", 0))
    drs4_dest_port3 = drs4_dest_port3 or int(getenv("DRS4_DEST_PORT3", 0))
    drs4_dest_port4 = drs4_dest_port4 or int(getenv("DRS4_DEST_PORT4", 0))

    with (
        Manager() as manager,
        ProcessPoolExecutor(4) as executor,
        TemporaryDirectory() as tempdir,
    ):
        vdif_in1 = Path(tempdir) / f"drs4-chassis{chassis}-in1-{obsid}.vdif"
        vdif_in2 = Path(tempdir) / f"drs4-chassis{chassis}-in2-{obsid}.vdif"
        vdif_in3 = Path(tempdir) / f"drs4-chassis{chassis}-in3-{obsid}.vdif"
        vdif_in4 = Path(tempdir) / f"drs4-chassis{chassis}-in4-{obsid}.vdif"

        cancel = manager.Event()
        executor.submit(dump, vdif_in1, drs4_dest_addr, drs4_dest_port1, cancel=cancel)
        executor.submit(dump, vdif_in2, drs4_dest_addr, drs4_dest_port2, cancel=cancel)
        executor.submit(dump, vdif_in3, drs4_dest_addr, drs4_dest_port3, cancel=cancel)
        executor.submit(dump, vdif_in4, drs4_dest_addr, drs4_dest_port4, cancel=cancel)

        try:
            sleep(int(duration))
        finally:
            cancel.set()

        ds_if1, ds_if2 = xr.align(
            open_vdifs(
                vdif_in1,
                vdif_in2,
                chassis=chassis,
                freq_range=freq_range_if1,
                integ_time=integ_time,
                interface=1,
            ),
            open_vdifs(
                vdif_in3,
                vdif_in4,
                chassis=chassis,
                freq_range=freq_range_if2,
                integ_time=integ_time,
                interface=2,
            ),
            join="inner",
        )

        ds_if1.to_zarr(zarr_if1, mode="w")
        ds_if2.to_zarr(zarr_if2, mode="w")
        return Path(zarr_if1).resolve(), Path(zarr_if2).resolve()
