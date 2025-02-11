__all__ = ["run"]


# standard library
from logging import getLogger
from os import PathLike, getenv
from subprocess import PIPE, CompletedProcess, run as sprun
from typing import Optional, Union, overload


# dependencies
from drs4.specs.zarr import Chassis


# type hints
StrCP = CompletedProcess[str]
StrPath = Union[PathLike[str], str]


# constants
ENV_CTRL_ADDR = "DRS4_CHASSIS{0}_CTRL_ADDR"
ENV_CTRL_USER = "DRS4_CHASSIS{0}_CTRL_USER"
LOGGER = getLogger(__name__)


@overload
def run(
    # for connection (required)
    *commands: str,
    # for connection (optional)
    chassis: Chassis,
    ctrl_addr: Optional[str] = None,
    ctrl_user: Optional[str] = None,
    timeout: Optional[float] = None,
) -> StrCP: ...


@overload
def run(
    # for connection (required)
    *commands: str,
    # for connection (optional)
    chassis: None = None,
    ctrl_addr: Optional[str] = None,
    ctrl_user: Optional[str] = None,
    timeout: Optional[float] = None,
) -> tuple[StrCP, StrCP]: ...


def run(
    # for connection (required)
    *commands: str,
    # for connection (optional)
    chassis: Optional[Chassis] = None,
    ctrl_addr: Optional[str] = None,
    ctrl_user: Optional[str] = None,
    timeout: Optional[float] = None,
    workdir: StrPath = "~/DRS4/cmd",
) -> Union[StrCP, tuple[StrCP, StrCP]]:
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

    script = ";".join((f"cd {workdir}", *commands))
    LOGGER.info(args := f"ssh {ctrl_user}@{ctrl_addr} '{script}'")

    result = sprun(
        args,
        shell=True,
        stderr=PIPE,
        stdout=PIPE,
        text=True,
        timeout=timeout,
    )

    if result.stdout:
        LOGGER.info(result.stdout)

    if result.stderr:
        LOGGER.error(result.stderr)

    return result
