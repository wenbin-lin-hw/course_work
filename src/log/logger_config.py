import logging
import os
from datetime import datetime
import sys


class LocalLogger:
    """Local logging system"""

    def __init__(self, log_dir="../../logs", log_prefix="log", log_level=logging.INFO):
        """
        Initialize logging system

        Args:
            log_dir: Log directory
            log_prefix: Log file prefix
            log_level: Logging level
        """
        self.log_dir = log_dir
        self.log_prefix = log_prefix
        self.log_level = log_level
        self.logger = None

        # Create log directory
        self._create_log_directory()

        # Setup logger
        self._setup_logger()

    def _create_log_directory(self):
        """Create log directory"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"Created log directory: {self.log_dir}")

    def _get_log_filename(self):
        """Generate log filename: log_YYYYMMDD_HHMMSS.log"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{self.log_prefix}_{timestamp}.log"

    def _setup_logger(self):
        """Setup logger configuration"""
        # Get log file path
        log_filename = self._get_log_filename()
        log_filepath = os.path.join(self.log_dir, log_filename)

        # Create logger
        self.logger = logging.getLogger(f"LocalLogger_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.logger.setLevel(self.log_level)

        # Prevent duplicate handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Create file handler
        file_handler = logging.FileHandler(log_filepath, mode='w', encoding='utf-8')
        file_handler.setLevel(self.log_level)

        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)

        # Create formatter
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

        # Set formatter
        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(console_formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Record log file information
        self.log_filepath = log_filepath
        print(f"Log file created: {log_filepath}")

    def get_logger(self):
        """Get logger instance"""
        return self.logger

    def get_log_filepath(self):
        """Get current log file path"""
        return self.log_filepath

    def info(self, message):
        """Log INFO level message"""
        self.logger.info(message)

    def debug(self, message):
        """Log DEBUG level message"""
        self.logger.debug(message)

    def warning(self, message):
        """Log WARNING level message"""
        self.logger.warning(message)

    def error(self, message):
        """Log ERROR level message"""
        self.logger.error(message)

    def critical(self, message):
        """Log CRITICAL level message"""
        self.logger.critical(message)


def setup_logger(script_name=None, log_level=logging.INFO):
    """
    Quick setup for logging system

    Args:
        script_name: Script name (used for log prefix)
        log_level: Logging level

    Returns:
        logger instance
    """
    if script_name is None:
        # Automatically get the calling script name
        import inspect
        frame = inspect.currentframe().f_back
        script_name = os.path.splitext(os.path.basename(frame.f_globals['__file__']))[0]

    # Create logging system
    local_logger = LocalLogger(
        log_dir="../../logs",
        log_prefix=f"log_{script_name}",
        log_level=log_level
    )

    return local_logger.get_logger()


# def get_logger():
#     """
#     Simple quick setup for logging system
#
#     Returns:
#         logger instance
#     """
#     return setup_logger()


