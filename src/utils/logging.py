"""
Logging utilities for experiments.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: str = None,
    name: str = None
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_file: Path to log file (if None, only console output)
        level: Logging level
        format_string: Custom format string
        name: Logger name (if None, uses root logger)

    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'

    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')

    # Console handler (minimal output - we use rich for main display)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)  # Only warnings and above to console
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (detailed logs)
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger by name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """
    Experiment logger that writes to both file and provides structured output.
    """

    def __init__(
        self,
        experiment_name: str,
        log_dir: str = 'logs',
        console_level: int = logging.WARNING
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'{experiment_name}_{timestamp}.log'

        # Set up logger
        self.logger = setup_logging(
            log_file=str(self.log_file),
            level=logging.DEBUG,
            name=experiment_name
        )

        # Adjust console level
        for handler in self.logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler.setLevel(console_level)

    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)

    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)

    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)

    def log_config(self, config: dict):
        """Log experiment configuration."""
        self.logger.info("=" * 60)
        self.logger.info("EXPERIMENT CONFIGURATION")
        self.logger.info("=" * 60)
        for key, value in config.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 60)

    def log_results(self, results: dict, title: str = "RESULTS"):
        """Log experiment results."""
        self.logger.info("=" * 60)
        self.logger.info(title)
        self.logger.info("=" * 60)
        self._log_dict(results, indent=2)
        self.logger.info("=" * 60)

    def _log_dict(self, d: dict, indent: int = 0):
        """Recursively log dictionary contents."""
        prefix = " " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                self.logger.info(f"{prefix}{key}:")
                self._log_dict(value, indent + 2)
            elif isinstance(value, float):
                self.logger.info(f"{prefix}{key}: {value:.4f}")
            else:
                self.logger.info(f"{prefix}{key}: {value}")

    def log_method_start(self, method_name: str):
        """Log start of method evaluation."""
        self.logger.info("-" * 40)
        self.logger.info(f"Starting evaluation: {method_name}")
        self.logger.info("-" * 40)

    def log_method_end(self, method_name: str, metrics: dict):
        """Log end of method evaluation with results."""
        self.logger.info(f"Completed: {method_name}")
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {metric}: {value:.4f}")
            else:
                self.logger.info(f"  {metric}: {value}")

    def log_fold(self, fold: int, num_folds: int):
        """Log fold information."""
        self.logger.info(f"Processing fold {fold}/{num_folds}")
