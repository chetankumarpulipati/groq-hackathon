"""
Logging configuration and utilities for the healthcare system.
"""

import os
import sys
from pathlib import Path
from loguru import logger
from config.settings import config


def setup_logging():
    """Configure logging for the entire system."""

    # Remove default logger
    logger.remove()

    # Create logs directory if it doesn't exist
    log_file_path = Path(config.logging.log_file)
    log_file_path.parent.mkdir(parents=True, exist_ok=True)

    # Console logging
    logger.add(
        sys.stdout,
        level=config.logging.log_level,  # Changed from 'level' to 'log_level'
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        colorize=True
    )

    # File logging
    logger.add(
        config.logging.log_file,
        level=config.logging.log_level,  # Changed from 'level' to 'log_level'
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        rotation="10 MB",  # Set default rotation
        retention="30 days",  # Set default retention
        compression="zip"
    )

    return logger


def get_logger(name: str):
    """Get a logger instance for a specific module."""
    return logger.bind(name=name)


# Initialize logging
setup_logging()
healthcare_logger = get_logger("healthcare_system")
