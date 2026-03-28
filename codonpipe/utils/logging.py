"""Logging configuration for CodonPipe."""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "codonpipe", log_file: Path | None = None, verbose: bool = False) -> logging.Logger:
    """Configure and return a logger instance.

    Args:
        name: Logger name.
        log_file: Optional path to write log file.
        verbose: If True, set DEBUG level; otherwise INFO.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
