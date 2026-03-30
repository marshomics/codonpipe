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

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add console handler only if one doesn't already exist
    has_stream_handler = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    if not has_stream_handler:
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(logging.DEBUG if verbose else logging.INFO)
        console.setFormatter(fmt)
        logger.addHandler(console)

    # Add file handler only if log_file is provided and no FileHandler for that path exists
    if log_file is not None:
        has_file_handler = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file)
            for h in logger.handlers
        )
        if not has_file_handler:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file, mode="a")
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger
