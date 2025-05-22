__all__ = ["gain"]


# standard library
from logging import getLogger
from subprocess import PIPE, run as sprun
from typing import Optional, Union, get_args


# dependencies
import xarray as xr
from ..ctrl.self import run
from ..specs.common import Chassis, Interface
from ..specs.gain import GAIN_ONES, GAIN_ZEROS, open_gain, to_dataframe
from ..utils import StrPath, is_strpath, set_workdir


# constants
LOGGER = getLogger(__name__)
PATH_COEF_TABLE = "~/DRS4/mrdsppy/coef_table/new_coef_table.csv"


def gain(
    ms: Optional[Union[xr.Dataset, StrPath]] = None,
    /,
    *,
    # for measurement (required)
    chassis: Optional[Chassis] = None,
    interface: Optional[Interface] = None,
    ones: bool = False,
    zeros: bool = False,
    # for connection (optional)
    ctrl_addr: Optional[str] = None,
    ctrl_user: Optional[str] = None,
    timeout: Optional[float] = None,
    workdir: Optional[StrPath] = None,
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
            gain(
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
            gain(
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

        result = sprun(
            f"scp {csv} {ctrl_user}@{ctrl_addr}:{PATH_COEF_TABLE}",
            shell=True,
            stderr=PIPE,
            stdout=PIPE,
            text=True,
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
