__all__ = ["auto", "autos", "cross", "crosses", "tcp", "udp"]


# submodules
from . import tcp
from . import udp

# aliases
from .tcp import cross, crosses
from .udp import auto, autos
