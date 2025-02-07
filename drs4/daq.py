__all__ = ["run"]


# standard library
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
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
from typing import Optional, Union, get_args


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
OBSID_FORMAT = "%Y%m%dT%H%M%SZ"


def dump(
    vdif: Union[Path, str],
    dest_addr: str,
    dest_port: int,
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


def run(
    *,
    # for file saving
    zarr_if1: Optional[StrPath] = None,
    zarr_if2: Optional[StrPath] = None,
    overwrite: bool = False,
    progress: bool = False,
    workdir: Optional[StrPath] = None,
    # for measurement
    chassis: Chassis = 1,
    duration: int = 3600,
    integ_time: IntegTime = 100,
    freq_range_if1: FreqRange = "inner",
    freq_range_if2: FreqRange = "outer",
    # for connection
    dest_addr: Optional[str] = None,
    dest_port1: Optional[int] = None,
    dest_port2: Optional[int] = None,
    dest_port3: Optional[int] = None,
    dest_port4: Optional[int] = None,
) -> tuple[Path, Path]:
    """"""
    obsid = datetime.now(timezone.utc).strftime(OBSID_FORMAT)

    if chassis not in get_args(Chassis):
        raise ValueError("Value of chassis must be 1|2.")

    if freq_range_if1 not in get_args(FreqRange):
        raise ValueError("Value of freq_range_if1 must be inner|outer.")

    if freq_range_if2 not in get_args(FreqRange):
        raise ValueError("Value of freq_range_if2 must be inner|outer.")

    if integ_time not in get_args(IntegTime):
        raise ValueError("Value of integ_time must be 100|200|500|1000.")

    if dest_addr is None:
        dest_addr = getenv(f"DRS4_CHASSIS{chassis}_DEST_ADDR", "")

    if dest_port1 is None:
        dest_port1 = int(getenv(f"DRS4_CHASSIS{chassis}_DEST_PORT1", ""))

    if dest_port2 is None:
        dest_port2 = int(getenv(f"DRS4_CHASSIS{chassis}_DEST_PORT2", ""))

    if dest_port3 is None:
        dest_port3 = int(getenv(f"DRS4_CHASSIS{chassis}_DEST_PORT3", ""))

    if dest_port4 is None:
        dest_port4 = int(getenv(f"DRS4_CHASSIS{chassis}_DEST_PORT4", ""))

    if zarr_if1 is None:
        zarr_if1 = f"drs4-{obsid}-chassis{chassis}-if1.zarr.zip"

    if zarr_if2 is None:
        zarr_if2 = f"drs4-{obsid}-chassis{chassis}-if2.zarr.zip"

    if Path(zarr_if1).exists() and not overwrite:
        raise FileExistsError(zarr_if1)

    if Path(zarr_if2).exists() and not overwrite:
        raise FileExistsError(zarr_if2)

    with (
        Manager() as manager,
        ProcessPoolExecutor(4) as executor,
        set_workdir(workdir) as workdir,
        tqdm(disable=not progress, total=int(duration), unit="s") as bar,
    ):
        cancel = manager.Event()
        vdif_in1 = Path(workdir) / f"drs4-chassis{chassis}-in1-{obsid}.vdif"
        vdif_in2 = Path(workdir) / f"drs4-chassis{chassis}-in2-{obsid}.vdif"
        vdif_in3 = Path(workdir) / f"drs4-chassis{chassis}-in3-{obsid}.vdif"
        vdif_in4 = Path(workdir) / f"drs4-chassis{chassis}-in4-{obsid}.vdif"
        executor.submit(dump, vdif_in1, dest_addr, dest_port1, cancel=cancel)
        executor.submit(dump, vdif_in2, dest_addr, dest_port2, cancel=cancel)
        executor.submit(dump, vdif_in3, dest_addr, dest_port3, cancel=cancel)
        executor.submit(dump, vdif_in4, dest_addr, dest_port4, cancel=cancel)

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


@contextmanager
def set_workdir(workdir: Optional[StrPath] = None, /) -> Iterator[Path]:
    """Set the working directory for output VDIF files."""
    if workdir is None:
        with TemporaryDirectory() as workdir:
            yield Path(workdir)
    else:
        yield Path(workdir).expanduser()
