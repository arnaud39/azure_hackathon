"""Init file of the package."""
import logging
from .logging_formatter import CustomFormatter
from .run import run

__all__ = ["run"]

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)
