"""Defines a custom logging formatter."""
import logging


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and customize messages."""

    blue = "\x1b[1;34m"
    grey = "\x1b[38;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    record_format = "[%(levelname)s] %(message)s"

    FORMATS = {
        logging.DEBUG: blue + record_format + reset,
        logging.INFO: grey + "%(message)s" + reset,
        logging.WARNING: yellow + record_format + reset,
        logging.ERROR: red + record_format + reset,
        logging.CRITICAL: bold_red + record_format + reset,
    }

    def format(self, record: logging.LogRecord) -> str:
        """Format the provided record."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
