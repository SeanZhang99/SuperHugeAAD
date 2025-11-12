from warnings import warn
from . import former, interface, linear, loss, module, pure_cnn
from .types import *

import os

try:
    from . import ssmamba
except ImportError as e:
    warn(
        f"Failed to import ssmamba module: {e}.\nThis does not cause your code fail to running, but ssmamba model is unavailable."
    )
