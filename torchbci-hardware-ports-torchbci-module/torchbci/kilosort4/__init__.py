#__version__ = "4"
import importlib.metadata
try:
    __version__ = importlib.metadata.version('kilosort')
except importlib.metadata.PackageNotFoundError:
    # package is not installed
    pass

from .utils import PROBE_DIR, DOWNLOADS_DIR
from .parameters import DEFAULT_SETTINGS
