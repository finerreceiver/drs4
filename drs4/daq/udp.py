__all__ = ["auto", "autos"]


# standard library
from collections.abc import Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
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
from threading import Barrier, BrokenBarrierError, Event
from time import sleep
from typing import Any
from warnings import filterwarnings

# dependencies
import xarray as xr
from tqdm import tqdm
from ..ctrl.self import run, set_gain
from ..specs.common import (
    ENV_CTRL_ADDR,
    ENV_CTRL_USER,
    ENV_DEST_ADDR,
    ENV_DEST_PORT1,
    ENV_DEST_PORT2,
    ENV_DEST_PORT3,
    ENV_DEST_PORT4,
    ZARR_CHUNKS,
    ZARR_ENCODING,
    ZARR_SHARDS,
    Channel,
    Chassis,
    DSPMode,
    FreqRange,
    Interface,
    IntegTime,
    SideBand,
)
from ..specs.ms import open_vdifs
from ..specs.vdif import VDIF_FRAME_BYTES
from ..utils import StrPath, XarrayJoin, is_strpath, set_workdir, unique

# global settings
GROUP = "239.0.0.1"
LOGGER = getLogger(__name__)
filterwarnings("ignore", category=FutureWarning)


def auto(
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
    dest_addr: str | None = None,
    dest_port1: int | None = None,
    dest_port2: int | None = None,
    dest_port3: int | None = None,
    dest_port4: int | None = None,
    dsp_mode: DSPMode = "SB",
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

    if isinstance(gain, xr.DataTree):
        gain_if1 = gain[f"/chassis{chassis}/if1"].to_dataset()
        gain_if2 = gain[f"/chassis{chassis}/if2"].to_dataset()
    elif is_strpath(gain):
        gain_if1 = Path(gain) / f"chassis{chassis}" / "if1"
        gain_if2 = Path(gain) / f"chassis{chassis}" / "if2"
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
            zeros=True if gain_if1 is None else False,
            ctrl_addr=ctrl_addr,
            ctrl_user=ctrl_user,
            timeout=timeout,
        )
        set_gain(
            gain_if2,
            chassis=chassis,
            interface=2,
            zeros=True if gain_if2 is None else False,
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
        Manager() as manager,
        ProcessPoolExecutor(4) as executor,
        set_workdir(workdir) as workdir,
        tqdm(
            desc=f"Chassis {chassis}",
            disable=not progress,
            leave=True,
            position=max(int(progress) - 1, 0),
            total=None if isinstance(duration, Event) else int(duration),
            unit="s",
        ) as bar,
    ):
        if sync is not None:
            try:
                sync.wait(timeout)
            except BrokenBarrierError:
                return zarr.resolve()

        bar.reset()
        interrupt = manager.Event()
        executor.submit(
            dump,
            vdif_in1 := workdir / f"{zarr.stem}-chassis{chassis}-in1.vdif",
            dest_addr=dest_addr,
            dest_port=dest_port1,
            interrupt=interrupt,
            overwrite=overwrite,
            timeout=timeout,
        )
        executor.submit(
            dump,
            vdif_in2 := workdir / f"{zarr.stem}-chassis{chassis}-in2.vdif",
            dest_addr=dest_addr,
            dest_port=dest_port2,
            interrupt=interrupt,
            overwrite=overwrite,
            timeout=timeout,
        )
        executor.submit(
            dump,
            vdif_in3 := workdir / f"{zarr.stem}-chassis{chassis}-in3.vdif",
            dest_addr=dest_addr,
            dest_port=dest_port3,
            interrupt=interrupt,
            overwrite=overwrite,
            timeout=timeout,
        )
        executor.submit(
            dump,
            vdif_in4 := workdir / f"{zarr.stem}-chassis{chassis}-in4.vdif",
            dest_addr=dest_addr,
            dest_port=dest_port4,
            interrupt=interrupt,
            overwrite=overwrite,
            timeout=timeout,
        )

        try:
            if isinstance(duration, Event):
                while not duration.wait(1.0):
                    bar.update(1)

                LOGGER.debug("Data acquisition finished by event.")
            else:
                for _ in range(int(duration)):
                    sleep(1)
                    bar.update(1)

                LOGGER.debug("Data acquisition finished by duration.")
        except KeyboardInterrupt:
            LOGGER.warning("Data acquisition interrupted by user.")
        finally:
            interrupt.set()

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

        if zarr.exists() and append:
            ds_if1.chunk(ZARR_CHUNKS).to_zarr(
                zarr,
                group=f"/chassis{chassis}/if1",
                mode="a",
                append_dim="time",
                consolidated=False,
                safe_chunks=False,
            )
            ds_if2.chunk(ZARR_CHUNKS).to_zarr(
                zarr,
                group=f"/chassis{chassis}/if2",
                mode="a",
                append_dim="time",
                consolidated=False,
                safe_chunks=False,
            )
        else:
            ds_if1.chunk(ZARR_CHUNKS).to_zarr(
                zarr,
                group=f"/chassis{chassis}/if1",
                mode="w",
                encoding=encoding_if1,
                consolidated=False,
                safe_chunks=False,
            )
            ds_if2.chunk(ZARR_CHUNKS).to_zarr(
                zarr,
                group=f"/chassis{chassis}/if2",
                mode="a",
                encoding=encoding_if2,
                consolidated=False,
                safe_chunks=False,
            )

        return zarr.resolve()


def autos(
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
    dsp_mode: DSPMode = "SB",
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
                auto,
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
                # for external synchronization (optional)
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


def dump(
    vdif: StrPath,
    /,
    *,
    dest_addr: str,
    dest_port: int,
    group: str = GROUP,
    interrupt: Event | None = None,
    overwrite: bool = False,
    progress: bool | int = False,
    timeout: float | None = None,
) -> None:
    """Receive and dump DRS4 data per input into a VDIF file.

    Args:
        vdif: Path of the output VDIF file.
        dest_addr: Destination IP address.
        dest_port: Destination port number.
        group: Multicast group IP address.
        interrupt: Event object to interrupt dumping.
        timeout: Timeout period in units of seconds.
        overwrite: Whether to overwrite the existing VDIF file.
        progress: Whether to show the progress bar on screen.

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
        tqdm(
            desc=prefix,
            disable=not progress,
            leave=True,
            position=max(int(progress) - 1, 0),
            unit="byte",
        ) as bar,
    ):
        # create socket
        sock.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
        sock.bind(("", dest_port))
        sock.setsockopt(IPPROTO_IP, IP_ADD_MEMBERSHIP, mreq)
        sock.settimeout(timeout)

        # start dumping
        LOGGER.debug(f"{prefix} Start dumping data.")

        while interrupt is None or not interrupt.is_set():
            frame, _ = sock.recvfrom(VDIF_FRAME_BYTES)

            if len(frame) == VDIF_FRAME_BYTES:
                file.write(frame)
                bar.update(VDIF_FRAME_BYTES)
            else:
                LOGGER.warning(f"{prefix} Truncated frame.")

        # finish dumping
        LOGGER.debug(f"{prefix} Finish dumping data.")
