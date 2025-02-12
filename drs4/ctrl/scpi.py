__all__ = ["connect", "send_commands", "send_commands_in"]


# standard library
from logging import getLogger
from socket import socket, AF_INET, SOCK_STREAM
from typing import IO, Optional, Sequence, Union


# dependencies
from ..utils import StrPath


# constants
DEFAULT_AUTORECV: bool = True
DEFAULT_BUFSIZE: int = 4096
DEFAULT_ENCODING: str = "ascii"
DEFAULT_END: str = "\n"
DEFAULT_FLAGS: int = 0
DEFAULT_TIMEOUT: Optional[float] = None
LOGGER = getLogger(__name__)


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
        LOGGER.info(f"{host}:{port} <- {string}")
        return n_bytes

    def recv(
        self,
        bufsize: int = DEFAULT_BUFSIZE,
        flags: int = DEFAULT_FLAGS,
        end: str = DEFAULT_END,
        encoding: str = DEFAULT_ENCODING,
    ) -> str:
        """Same as socket.recv(), but returns string, not bytes."""
        received = super().recv(bufsize, flags)
        string = received.decode(encoding).rstrip(end)

        host, port = self.getpeername()
        LOGGER.info(f"{host}:{port} -> {string}")
        return string


def connect(
    host: str, port: int, timeout: Optional[float] = DEFAULT_TIMEOUT
) -> CustomSocket:
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
    sock = CustomSocket(AF_INET, SOCK_STREAM)
    sock.settimeout(timeout)
    sock.connect((host, port))

    return sock


def send_commands(
    commands: Union[IO[str], Sequence[str], str],
    host: str,
    port: int,
    timeout: Optional[float] = DEFAULT_TIMEOUT,
    encoding: str = DEFAULT_ENCODING,
    autorecv: bool = DEFAULT_AUTORECV,
    bufsize: int = DEFAULT_BUFSIZE,
) -> None:
    """Send SCPI command(s) to a server.

    Args:
        commands: Sequence of SCPI commands.
        host: IP address or host name of the server.
        port: Port of the server.
        timeout: Timeout value in units of seconds.
        encoding: Encoding format for the commands.
        autorecv: If True and a command ends with '?',
            receive a message and record it to a logger.
        bufsize: Maximum byte size for receiving a message.

    Returns:
        This function returns nothing.

    Examples:
        To send an SCPI command to the server::

            send_commands('*CLS', '192.168.1.3', 5000)

        To send SCPI commands to the server::

            send_commands(['*RST', '*CLS'], '192.168.1.3', 5000)

    """
    if isinstance(commands, str):
        commands = (commands,)

    with connect(host, port, timeout) as sock:
        for command in commands:
            if not command or command.startswith("#"):
                continue

            sock.send(command.strip(), encoding=encoding)

            if autorecv and command.endswith("?"):
                sock.recv(bufsize)


def send_commands_in(
    path: StrPath,
    host: str,
    port: int,
    timeout: Optional[float] = DEFAULT_TIMEOUT,
    encoding: str = DEFAULT_ENCODING,
    autorecv: bool = DEFAULT_AUTORECV,
    bufsize: int = DEFAULT_BUFSIZE,
) -> None:
    """Send SCPI command(s) written in a file to a server.

    Args:
        path: Path of the file.
        port: Port of the server.
        timeout: Timeout value in units of seconds.
        encoding: Encoding format for the commands.
        autorecv: If True and a command ends with '?',
            receive a message and record it to a logger.
        bufsize: Maximum byte size for receiving a message.

    Returns:
        This function returns nothing.

    Examples:
        If a text file, commands.txt, has SCPI commands::

            *RST
            *CLS

        then the following two commands are equivalent::

            send_commands(['*RST', '*CLS'], '192.168.1.3', 5000)
            send_commands_in('commands.txt', '192.168.1.3', 5000)

    """
    with open(path, encoding=encoding) as f:
        send_commands(f, host, port, timeout, encoding, autorecv, bufsize)
