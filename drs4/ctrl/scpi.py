__all__ = ["connect", "send_commands", "send_commands_in"]


# standard library
from logging import getLogger
from serial import Serial
from socket import socket, AF_INET, SOCK_STREAM
from typing import Any, IO, Sequence, overload

# dependencies
from ..utils import StrPath

# constants
DEFAULT_AUTORECV: bool = True
DEFAULT_BAUDRATE: int = 9600
DEFAULT_BUFSIZE: int = 4096
DEFAULT_ENCODING: str = "ascii"
DEFAULT_END: str = "\n"
DEFAULT_FLAGS: int = 0
DEFAULT_TIMEOUT: float | None = None
LOGGER = getLogger(__name__)


class CustomSerial(Serial):
    """Custom serial class to send/recv string with logging."""

    def send(
        self,
        string: str,
        flags: int = DEFAULT_FLAGS,
        end: str = DEFAULT_END,
        encoding: str = DEFAULT_ENCODING,
    ) -> int:
        """Same as Serial.write(), but accepts string, not bytes."""
        encoded = (string + end).encode(encoding)

        if (n_bytes := self.write(encoded)) is None:
            raise ConnectionError("Unexpected disconnect.")

        LOGGER.debug(f"{self.port} <- {string}")
        return n_bytes

    def recv(
        self,
        bufsize: int = DEFAULT_BUFSIZE,
        flags: int = DEFAULT_FLAGS,
        end: str = DEFAULT_END,
        encoding: str = DEFAULT_ENCODING,
    ) -> str:
        """Same as Serial.read_until(), but returns string, not bytes."""
        bend = end.encode(encoding)
        received = self.read_until(expected=bend)

        if not received or not received.endswith(bend):
            raise TimeoutError("Unexpected disconnect.")

        string = received.decode(encoding).removesuffix(end)
        LOGGER.debug(f"{self.port} -> {string}")
        return string


class CustomSocket(socket):
    """Custom socket class to send/recv string with logging."""

    def send(
        self,
        string: str,
        flags: int = DEFAULT_FLAGS,
        end: str = DEFAULT_END,
        encoding: str = DEFAULT_ENCODING,
    ) -> int:
        """Same as socket.send(), but accepts string, not bytes."""
        encoded = (string + end).encode(encoding)
        n_bytes = super().send(encoded, flags)

        host, port = self.getpeername()
        LOGGER.debug(f"{host}:{port} <- {string}")
        return n_bytes

    def recv(
        self,
        bufsize: int = DEFAULT_BUFSIZE,
        flags: int = DEFAULT_FLAGS,
        end: str = DEFAULT_END,
        encoding: str = DEFAULT_ENCODING,
    ) -> str:
        """Same as socket.recv(), but returns string, not bytes."""
        bend = end.encode(encoding)
        received = bytearray()

        while True:
            if not (packet := super().recv(bufsize, flags)):
                raise ConnectionError("Unexpected disconnect.")

            received.extend(packet)

            if received.endswith(bend):
                break

        string = received.decode(encoding).removesuffix(end)
        host, port = self.getpeername()
        LOGGER.debug(f"{host}:{port} -> {string}")
        return string


@overload
def connect(
    host: None,
    port: str,
    /,
    *,
    timeout: float | None = DEFAULT_TIMEOUT,
) -> CustomSerial: ...
@overload
def connect(
    host: str,
    port: int,
    /,
    *,
    timeout: float | None = DEFAULT_TIMEOUT,
) -> CustomSocket: ...
def connect(
    host: Any,
    port: Any,
    /,
    *,
    timeout: float | None = DEFAULT_TIMEOUT,
) -> Any:
    """Connect to an SCPI server and returns a custom socket object.

    Args:
        host: IP address or host name of the server.
        port: Port of the server.
        timeout: Timeout value in units of seconds.

    Returns:
        Custom socket object.

    Examples:
        To send an SCPI command to a server::

            with connect('192.168.1.3', 5000) as sock:
                sock.send('*CLS')

        To receive a message from a server::

            with connect('192.168.1.3', 5000) as sock:
                sock.send('SYST:ERR?')
                print(sock.recv())

    """
    LOGGER.debug("(")

    for key, val in locals().items():
        LOGGER.debug(f"  {key}: {val!r}")

    LOGGER.debug(")")

    if host is None and isinstance(port, str):
        return CustomSerial(
            port,
            baudrate=DEFAULT_BAUDRATE,
            timeout=timeout,
        )

    if host is not None and isinstance(port, int):
        conn = CustomSocket(AF_INET, SOCK_STREAM)
        conn.settimeout(timeout)
        conn.connect((host, port))
        return conn

    raise ValueError("Invalid host or port.")


@overload
def send_commands(
    commands: IO[str] | Sequence[str] | str,
    /,
    *,
    host: None,
    port: str,
    timeout: float | None = DEFAULT_TIMEOUT,
    encoding: str = DEFAULT_ENCODING,
    autorecv: bool = DEFAULT_AUTORECV,
    bufsize: int = DEFAULT_BUFSIZE,
) -> tuple[str, ...]: ...
@overload
def send_commands(
    commands: IO[str] | Sequence[str] | str,
    /,
    *,
    host: str,
    port: int,
    timeout: float | None = DEFAULT_TIMEOUT,
    encoding: str = DEFAULT_ENCODING,
    autorecv: bool = DEFAULT_AUTORECV,
    bufsize: int = DEFAULT_BUFSIZE,
) -> tuple[str, ...]: ...
def send_commands(
    commands: IO[str] | Sequence[str] | str,
    /,
    *,
    host: Any,
    port: Any,
    timeout: float | None = DEFAULT_TIMEOUT,
    encoding: str = DEFAULT_ENCODING,
    autorecv: bool = DEFAULT_AUTORECV,
    bufsize: int = DEFAULT_BUFSIZE,
) -> tuple[str, ...]:
    """Send SCPI command(s) to a server.

    Args:
        commands: Sequence of SCPI commands.
        host: IP address or host name of the server.
        port: Port of the server.
        timeout: Timeout value in units of seconds.
        encoding: Encoding format for the commands.
        autorecv: If True and a command contains '?',
            receive a message and record it to a logger.
        bufsize: Maximum byte size for receiving a message.

    Returns:
        Tuple of the received messages.

    Examples:
        To send an SCPI command to the server::

            send_commands('*CLS', '192.168.1.3', 5000)

        To send SCPI commands to the server::

            send_commands(['*RST', '*CLS'], '192.168.1.3', 5000)

    """
    LOGGER.debug("(")

    for key, val in locals().items():
        LOGGER.debug(f"  {key}: {val!r}")

    LOGGER.debug(")")

    if isinstance(commands, str):
        commands = (commands,)

    with connect(host, port, timeout=timeout) as conn:
        messages: list[str] = []

        for command in commands:
            if not command or command.startswith("#"):
                continue

            conn.send(command.strip(), encoding=encoding)

            if autorecv and "?" in command:
                messages.append(conn.recv(bufsize))

        return tuple(messages)


@overload
def send_commands_in(
    path: StrPath,
    /,
    *,
    host: None,
    port: str,
    timeout: float | None = DEFAULT_TIMEOUT,
    encoding: str = DEFAULT_ENCODING,
    autorecv: bool = DEFAULT_AUTORECV,
    bufsize: int = DEFAULT_BUFSIZE,
) -> tuple[str, ...]: ...
@overload
def send_commands_in(
    path: StrPath,
    /,
    *,
    host: str,
    port: int,
    timeout: float | None = DEFAULT_TIMEOUT,
    encoding: str = DEFAULT_ENCODING,
    autorecv: bool = DEFAULT_AUTORECV,
    bufsize: int = DEFAULT_BUFSIZE,
) -> tuple[str, ...]: ...
def send_commands_in(
    path: StrPath,
    /,
    *,
    host: Any,
    port: Any,
    timeout: float | None = DEFAULT_TIMEOUT,
    encoding: str = DEFAULT_ENCODING,
    autorecv: bool = DEFAULT_AUTORECV,
    bufsize: int = DEFAULT_BUFSIZE,
) -> tuple[str, ...]:
    """Send SCPI command(s) written in a file to a server.

    Args:
        path: Path of the file.
        port: Port of the server.
        timeout: Timeout value in units of seconds.
        encoding: Encoding format for the commands.
        autorecv: If True and a command contains '?',
            receive a message and record it to a logger.
        bufsize: Maximum byte size for receiving a message.

    Returns:
        Tuple of the received messages.

    Examples:
        If a text file, commands.txt, has SCPI commands::

            *RST
            *CLS

        then the following two commands are equivalent::

            send_commands(['*RST', '*CLS'], '192.168.1.3', 5000)
            send_commands_in('commands.txt', '192.168.1.3', 5000)

    """
    LOGGER.debug("(")

    for key, val in locals().items():
        LOGGER.debug(f"  {key}: {val!r}")

    LOGGER.debug(")")

    with open(path, encoding=encoding) as f:
        return send_commands(
            f,
            host=host,
            port=port,
            timeout=timeout,
            encoding=encoding,
            autorecv=autorecv,
            bufsize=bufsize,
        )
