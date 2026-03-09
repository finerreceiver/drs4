__all__ = ["run", "send", "set_gain"]


# standard library
from logging import getLogger
from os import getenv
from subprocess import PIPE, CompletedProcess, run as sprun
from typing import get_args, overload

# dependencies
import xarray as xr
from ..specs.common import ENV_CTRL_ADDR, ENV_CTRL_USER, Chassis, Interface
from ..specs.gain import GAIN_ONES, GAIN_ZEROS, open_gain, to_dataframe
from ..utils import StrPath, is_strpath, set_workdir

# type hints
StrCP = CompletedProcess[str]


# constants
LOGGER = getLogger(__name__)
PATH_CMD_DIR = "~/DRS4/cmd"
PATH_COEF_TABLE = "~/DRS4/mrdsppy/coef_table/new_coef_table.csv"


@overload
def run(
    *commands: str,
    chassis: Chassis,
    ctrl_addr: str | None = None,
    ctrl_user: str | None = None,
    timeout: float | None = None,
    workdir: StrPath = PATH_CMD_DIR,
) -> StrCP: ...


@overload
def run(
    *commands: str,
    chassis: None = None,
    ctrl_addr: str | None = None,
    ctrl_user: str | None = None,
    timeout: float | None = None,
    workdir: StrPath = PATH_CMD_DIR,
) -> tuple[StrCP, StrCP]: ...


def run(
    # for connection (required)
    *commands: str,
    # for connection (optional)
    chassis: Chassis | None = None,
    ctrl_addr: str | None = None,
    ctrl_user: str | None = None,
    timeout: float | None = None,
    workdir: StrPath = PATH_CMD_DIR,
) -> StrCP | tuple[StrCP, StrCP]:
    """Run commands in DRS4.

    Args:
        commands: Strings of the commands.
        chassis: Chassis number of DRS4 (1|2).
            If not specified, the commands will be run in both chasses.
        ctrl_addr: IP address of DRS4. If not specified,
            environment variable ``DRS4_CHASSIS[1|2]_CTRL_ADDR`` will be used.
        ctrl_user: User name of DRS4. If not specified,
            environment variable ``DRS4_CHASSIS[1|2]_CTRL_USER`` will be used.
        timeout: Timeout of the connection and the running in seconds.
        workdir: Working directory where commands will be run.

    Returns:
        Completed process object(s) of the run(s).

    Examples:
        To set the spectral integration time of DRS4 chassis 1::

            run("./set_intg_time.py --In 1 --It 1", chassis=1)

        To set the spectral integration time of DRS4 chassis 1 and 2::

            run("./set_intg_time.py --In 1 --It 1")

    """
    if chassis is None:
        return (
            run(
                *commands,
                chassis=1,
                ctrl_addr=ctrl_addr,
                ctrl_user=ctrl_user,
                timeout=timeout,
            ),
            run(
                *commands,
                chassis=2,
                ctrl_addr=ctrl_addr,
                ctrl_user=ctrl_user,
                timeout=timeout,
            ),
        )

    if ctrl_addr is None:
        ctrl_addr = getenv(ENV_CTRL_ADDR.format(chassis), "")

    if ctrl_user is None:
        ctrl_user = getenv(ENV_CTRL_USER.format(chassis), "")

    LOGGER.debug("(")

    for key, val in locals().items():
        LOGGER.debug(f"  {key}: {val!r}")

    LOGGER.debug(")")

    script = ";".join((f"cd {workdir}", *commands))
    args = f"ssh {ctrl_user}@{ctrl_addr} '{script}'"
    LOGGER.debug(args)

    result = sprun(
        args,
        shell=True,
        stderr=PIPE,
        stdout=PIPE,
        text=True,
        timeout=timeout,
    )

    if result.stdout:
        for line in result.stdout.split("\n"):
            LOGGER.debug(line)

    if result.stderr:
        for line in result.stderr.split("\n"):
            LOGGER.error(line)

    return result


@overload
def send(
    file: StrPath,
    to: StrPath,
    /,
    *,
    chassis: Chassis,
    ctrl_addr: str | None = None,
    ctrl_user: str | None = None,
    timeout: float | None = None,
) -> StrCP: ...


@overload
def send(
    file: StrPath,
    to: StrPath,
    /,
    *,
    chassis: None = None,
    ctrl_addr: str | None = None,
    ctrl_user: str | None = None,
    timeout: float | None = None,
) -> tuple[StrCP, StrCP]: ...


def send(
    # for file sending (required)
    file: StrPath,
    to: StrPath,
    /,
    *,
    # for connection (optional)
    chassis: Chassis | None = None,
    ctrl_addr: str | None = None,
    ctrl_user: str | None = None,
    timeout: float | None = None,
) -> StrCP | tuple[StrCP, StrCP]:
    """Send a file to DRS4.

    Args:
        file: Path of the file to be sent.
        to: Path of the destination in DRS4.
        chassis: Chassis number of DRS4 (1|2).
            If not specified, the file will be sent to both chasses.
        ctrl_addr: IP address of DRS4. If not specified,
            environment variable ``DRS4_CHASSIS[1|2]_CTRL_ADDR`` will be used.
        ctrl_user: User name of DRS4. If not specified,
            environment variable ``DRS4_CHASSIS[1|2]_CTRL_USER`` will be used.
        timeout: Timeout of the connection and the sending in seconds.

    Returns:
        Completed process object(s) of the sending(s).

    """
    if chassis is None:
        return (
            send(
                file,
                to,
                chassis=1,
                ctrl_addr=ctrl_addr,
                ctrl_user=ctrl_user,
                timeout=timeout,
            ),
            send(
                file,
                to,
                chassis=2,
                ctrl_addr=ctrl_addr,
                ctrl_user=ctrl_user,
                timeout=timeout,
            ),
        )

    if ctrl_addr is None:
        ctrl_addr = getenv(ENV_CTRL_ADDR.format(chassis), "")

    if ctrl_user is None:
        ctrl_user = getenv(ENV_CTRL_USER.format(chassis), "")

    LOGGER.debug("(")

    for key, val in locals().items():
        LOGGER.debug(f"  {key}: {val!r}")

    LOGGER.debug(")")

    result = sprun(
        f"scp {file} {ctrl_user}@{ctrl_addr}:{to}",
        shell=True,
        stderr=PIPE,
        stdout=PIPE,
        text=True,
        timeout=timeout,
    )

    if result.stdout:
        for line in result.stdout.split("\n"):
            LOGGER.debug(line)

    if result.stderr:
        for line in result.stderr.split("\n"):
            LOGGER.error(line)

    return result


def set_gain(
    ms: xr.Dataset | StrPath | None = None,
    /,
    *,
    # for measurement (required)
    chassis: Chassis | None = None,
    interface: Interface | None = None,
    ones: bool = False,
    zeros: bool = False,
    # for connection (optional)
    ctrl_addr: str | None = None,
    ctrl_user: str | None = None,
    timeout: float | None = None,
    workdir: StrPath | None = None,
) -> None:
    """Set a gain file (DRS4 MS file) to DRS4.

    Args:
        ms: Path of input gain file (DRS4 MS file) or gain Dataset itself.
        chassis: Chassis number of DRS4 (1|2).
            If not specified, the gain file will be applied to both chasses.
        interface: Interface number of DRS4 (1|2).
            If not specified, the gain file will be applied to both interfaces.
        zeros: If True, the zero-filled gain will be applied.
            It cannot be used with an input gain file nor ``ones``.
        ones: If True, the one-filled gain will be applied.
            It cannot be used with an input gain file nor ``zeros``.
        ctrl_addr: IP address of DRS4. If not specified,
            environment variable ``DRS4_CHASSIS[1|2]_CTRL_ADDR`` will be used.
        ctrl_user: User name of DRS4. If not specified,
            environment variable ``DRS4_CHASSIS[1|2]_CTRL_USER`` will be used.
        timeout: Timeout of the connection and the running in seconds.
        workdir: Working directory where an intermediate file will be put.

    """
    if chassis is None:
        for chassis in get_args(Chassis):
            set_gain(
                ms,
                chassis=chassis,
                interface=interface,
                ones=ones,
                zeros=zeros,
                ctrl_addr=ctrl_addr,
                ctrl_user=ctrl_user,
                timeout=timeout,
                workdir=workdir,
            )

        return

    if interface is None:
        for interface in get_args(Interface):
            set_gain(
                ms,
                chassis=chassis,
                interface=interface,
                ones=ones,
                zeros=zeros,
                ctrl_addr=ctrl_addr,
                ctrl_user=ctrl_user,
                timeout=timeout,
                workdir=workdir,
            )

        return

    LOGGER.info("(")

    for key, val in locals().items():
        LOGGER.info(f"  {key}: {val!r}")

    LOGGER.info(")")

    if chassis not in get_args(Chassis):
        raise ValueError("Chassis number must be 1|2.")

    if interface not in get_args(Interface):
        raise ValueError("Interface number must be 1|2.")

    if isinstance(ms, xr.Dataset) and not ones and not zeros:
        ds = ms
    elif is_strpath(ms) and not ones and not zeros:
        ds = open_gain(ms)
    elif ms is None and ones and not zeros:
        ds = GAIN_ONES
    elif ms is None and not ones and zeros:
        ds = GAIN_ZEROS
    else:
        raise ValueError("Either ms, ones, or zeros must be given.")

    with set_workdir(workdir) as workdir:
        to_dataframe(ds).to_csv(csv := workdir / "coef_table.csv")

        result = send(
            csv,
            PATH_COEF_TABLE,
            chassis=chassis,
            ctrl_addr=ctrl_addr,
            ctrl_user=ctrl_user,
            timeout=timeout,
        )
        result.check_returncode()

        result = run(
            f"./set_coef_tbl.py --In {1 if interface == 1 else 3}",
            chassis=chassis,
            ctrl_addr=ctrl_addr,
            ctrl_user=ctrl_user,
            timeout=timeout,
        )
        result.check_returncode()
