"""Centralized logging configuration for vbhs scripts.

This module provides utilities to configure logging based on verbosity levels,
similar to argparse's -v, -vv pattern.
"""

import logging


def configure_logging(verbosity: int = 0) -> None:
    """Configure logging based on verbosity level.

    Args:
        verbosity: Verbosity level (0 = WARNING, 1 = INFO, 2+ = DEBUG)
    """
    if verbosity == 0:
        log_level = logging.WARNING  # Default: only warnings and errors
    elif verbosity == 1:
        log_level = logging.INFO  # -v: show info messages
    else:  # verbosity >= 2
        log_level = logging.DEBUG  # -vv: show debug messages

    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s][%(levelname)s][%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

