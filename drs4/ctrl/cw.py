__all__ = ["on", "off", "status"]


# standard library
from logging import getLogger
from os import getenv
from typing import Optional


# constants
LOGGER = getLogger(__name__)


# dependencies
from .scpi import send_commands
from ..specs.common import (
    CHAN_TOTAL,
    FREQ_INTERVAL,
    ENV_LO_FREQ,
    ENV_LO_MULT,
    ENV_SG_ADDR,
    ENV_SG_AMPL,
    ENV_SG_PORT,
    Channel,
    FreqRange,
    SideBand,
)


def on(
    *,
    # for measurement (required)
    freq_range: FreqRange,
    signal_sb: SideBand,
    signal_chan: Channel,
    # for measurement (optional)
    lo_freq: Optional[float] = None,
    lo_mult: Optional[int] = None,
    sg_ampl: Optional[float] = None,
    # for connection (optional)
    sg_host: Optional[str] = None,
    sg_port: Optional[int] = None,
    timeout: Optional[float] = None,
) -> None:
    """Start outputting the CW signal after setting the SG amplitude and frequency.

    Args:
        freq_range: Intermediate frequency range (inner|outer).
        signal_chan: Signal channel number (0-511).
        signal_sb: Signal sideband (USB|LSB).
        lo_freq: LO frequency in GHz.
            If not specified, environment variable ``DRS4_LO_FREQ`` will be used.
        lo_mult: LO multiplication factor.
            If not specified, environment variable ``DRS4_LO_MULT`` will be used.
        sg_ampl: Amplitude of the CW signal in dBm.
            If not specified, environment variable ``DRS4_CW_SG_AMPL`` will be used.
        sg_host: Host name or IP address of the SG (e.g. Keysight 8257D).
            If not specified, environment variable ``DRS4_CW_SG_HOST`` will be used.
        sg_port: Port number of the SG (e.g. Keysight 8257D).
            If not specified, environment variable ``DRS4_CW_SG_PORT`` will be used.
        timeout: Timeout of the connection process in seconds.

    """
    if lo_freq is None:
        lo_freq = float(getenv(ENV_LO_FREQ, 0.0))

    if lo_mult is None:
        lo_mult = int(getenv(ENV_LO_MULT, 0))

    if sg_ampl is None:
        sg_ampl = float(getenv(ENV_SG_AMPL, 0.0))

    if sg_host is None:
        sg_host = str(getenv(ENV_SG_ADDR, ""))

    if sg_port is None:
        sg_port = int(getenv(ENV_SG_PORT, 0))

    if freq_range == "outer":
        signal_chan = CHAN_TOTAL * 2 - signal_chan

    if signal_sb == "USB":
        sg_freq = (lo_freq + FREQ_INTERVAL * signal_chan) / lo_mult
    elif signal_sb == "LSB":
        sg_freq = (lo_freq - FREQ_INTERVAL * signal_chan) / lo_mult
    else:
        raise ValueError("Signal sideband must be USB|LSB.")

    send_commands(
        [
            "OUTP OFF",
            "FREQ:MODE CW",
            f"FREQ:CW {sg_freq}GHz",
            f"AMPL {sg_ampl}dBm",
            "OUTP ON",
        ],
        host=sg_host,
        port=sg_port,
        timeout=timeout,
    )


def off(
    *,
    # for connection (optional)
    sg_host: Optional[str] = None,
    sg_port: Optional[int] = None,
    timeout: Optional[float] = None,
) -> None:
    """Stop outputting the CW signal.

    Args:
        sg_host: Host name or IP address of the SG (e.g. Keysight 8257D).
            If not specified, environment variable ``DRS4_CW_SG_HOST`` will be used.
        sg_port: Port number of the SG (e.g. Keysight 8257D).
            If not specified, environment variable ``DRS4_CW_SG_PORT`` will be used.
        timeout: Timeout of the connection process in seconds.

    """
    if sg_host is None:
        sg_host = getenv(ENV_SG_ADDR, "")

    if sg_port is None:
        sg_port = int(getenv(ENV_SG_PORT, ""))

    send_commands(
        "OUTP OFF",
        host=sg_host,
        port=sg_port,
        timeout=timeout,
    )


def status(
    *,
    # for connection (optional)
    sg_host: Optional[str] = None,
    sg_port: Optional[int] = None,
    timeout: Optional[float] = None,
) -> None:
    """Show the status of CW signal in the logger.

    Args:
        sg_host: Host name or IP address of the SG (e.g. Keysight 8257D).
            If not specified, environment variable ``DRS4_CW_SG_HOST`` will be used.
        sg_port: Port number of the SG (e.g. Keysight 8257D).
            If not specified, environment variable ``DRS4_CW_SG_PORT`` will be used.
        timeout: Timeout of the connection process in seconds.

    """
    if sg_host is None:
        sg_host = getenv(ENV_SG_ADDR, "")

    if sg_port is None:
        sg_port = int(getenv(ENV_SG_PORT, ""))

    send_commands(
        ["AMPL?", "FREQ?", "OUTP?"],
        host=sg_host,
        port=sg_port,
        timeout=timeout,
    )
