__all__ = ["run"]


# standard library
from datetime import datetime
from logging import getLogger
from os import getenv
from os.path import getmtime
from pathlib import Path
from subprocess import PIPE, run as sprun
from typing import Optional


# dependencies
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from PIL import Image
from ..specs.common import ENV_CTRL_ADDR, ENV_CTRL_USER, Chassis
from ..utils import StrPath, set_workdir


# constants
IMAGE_DIR = "~/DRS4"
IMAGE_00 = Path("disp_graph_1.jpg")
IMAGE_01 = Path("disp_histogram1.jpg")
IMAGE_02 = Path("disp_histogram2.jpg")
IMAGE_10 = Path("disp_graph_2.jpg")
IMAGE_11 = Path("disp_histogram3.jpg")
IMAGE_12 = Path("disp_histogram4.jpg")
LOGGER = getLogger(__name__)


def run(
    *,
    # for connection (required)
    chassis: Chassis,
    # for connection (optional)
    ctrl_addr: Optional[str] = None,
    ctrl_user: Optional[str] = None,
    timeout: Optional[float] = None,
    # for plotting (optional)
    figsize: Optional[tuple[int, int]] = (12, 6),
    interval: int = 10,
    workdir: Optional[StrPath] = None,
) -> None:
    """Display current auto-spectra and bit distributions of DRS4.

    Args:
        chassis: Chassis number of DRS4 (1|2).
        ctrl_addr: IP address of DRS4. If not specified,
            environment variable ``DRS4_CHASSIS[1|2]_CTRL_ADDR`` will be used.
        ctrl_user: User name of DRS4. If not specified,
            environment variable ``DRS4_CHASSIS[1|2]_CTRL_USER`` will be used.
        timeout: Timeout of the connection process in seconds.
        figsize: Width and height of the display in inches.
        interval: Refresh interval of the display in seconds.
        workdir: Working directory where intermediate images will be put.

    """
    if ctrl_addr is None:
        ctrl_addr = getenv(ENV_CTRL_ADDR.format(chassis), "")

    if ctrl_user is None:
        ctrl_user = getenv(ENV_CTRL_USER.format(chassis), "")

    LOGGER.debug("(")

    for key, val in locals().items():
        LOGGER.info(f"  {key}: {val!r}")

    LOGGER.debug(")")

    with set_workdir(workdir) as workdir:
        fig, axes = plt.subplots(2, 3, figsize=figsize)

        def plot(frame: int, /) -> None:
            sprun(
                f"scp -p '{ctrl_user}@{ctrl_addr}:{IMAGE_DIR}/*.jpg' {workdir}",
                stdout=PIPE,
                stderr=PIPE,
                shell=True,
                timeout=timeout,
            )

            for ax in axes.flatten():
                ax.cla()

            axes[0, 0].imshow(Image.open(workdir / IMAGE_00))
            axes[0, 1].imshow(Image.open(workdir / IMAGE_01))
            axes[0, 2].imshow(Image.open(workdir / IMAGE_02))
            axes[1, 0].imshow(Image.open(workdir / IMAGE_10))
            axes[1, 1].imshow(Image.open(workdir / IMAGE_11))
            axes[1, 2].imshow(Image.open(workdir / IMAGE_12))

            axes[0, 1].text(0, 100, "Bit Distribution (Port 1)", fontsize=15)
            axes[0, 2].text(0, 100, "Bit Distribution (Port 2)", fontsize=15)
            axes[1, 1].text(0, 100, "Bit Distribution (Port 3)", fontsize=15)
            axes[1, 2].text(0, 100, "Bit Distribution (Port 4)", fontsize=15)

            axes[0, 0].set_title(f"Last Updated: {mtime(workdir / IMAGE_00)}")
            axes[0, 1].set_title(f"Last Updated: {mtime(workdir / IMAGE_01)}")
            axes[0, 2].set_title(f"Last Updated: {mtime(workdir / IMAGE_02)}")
            axes[1, 0].set_title(f"Last Updated: {mtime(workdir / IMAGE_10)}")
            axes[1, 1].set_title(f"Last Updated: {mtime(workdir / IMAGE_11)}")
            axes[1, 2].set_title(f"Last Updated: {mtime(workdir / IMAGE_12)}")

            for ax in axes.flatten():
                ax.axis("off")

    try:
        plot(-1)
        fig.tight_layout()

        animation = FuncAnimation(  # type: ignore
            fig=fig,
            func=plot,  # type: ignore
            interval=interval * 1000,  # ms
            blit=False,
            cache_frame_data=False,
        )
        plt.show()
    except KeyboardInterrupt:
        LOGGER.warning("Plotting interrupted by user.")
    finally:
        plt.close()


def mtime(file: StrPath, /) -> str:
    """Return the modified datetime (YYYY-mm-ddTHH:MM:SS) of given file."""
    return datetime.fromtimestamp(getmtime(file)).isoformat(timespec="seconds")
