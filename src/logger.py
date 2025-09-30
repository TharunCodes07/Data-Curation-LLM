"""
Logging configuration for the data curation pipeline.
"""
import os
import sys
from pathlib import Path
from loguru import logger
from src.config import config


def setup_logging():
    """Setup logging configuration."""
    # Remove default handler
    logger.remove()
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Get logging config
    log_level = config.get('logging.level', 'INFO')
    log_format = config.get('logging.format', 
                           "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}")
    log_file = config.get('logging.log_file', './logs/app.log')
    
    # Add console handler
    logger.add(
        sys.stdout,
        format=log_format,
        level=log_level,
        colorize=True
    )
    
    # Add file handler
    logger.add(
        log_file,
        format=log_format,
        level=log_level,
        rotation="10 MB",
        retention="7 days",
        compression="zip"
    )
    
    return logger


# Initialize logging
app_logger = setup_logging()