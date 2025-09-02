import logging
import logging.handlers
import os
import sys
from pathlib import Path
from .constants import DEBUG

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }

    def format(self, record):
        # Create a copy to avoid modifying the original record
        record_copy = logging.makeLogRecord(record.__dict__)
        log_color = self.COLORS.get(record_copy.levelname, self.COLORS['RESET'])
        record_copy.levelname = f"{log_color}{record_copy.levelname}{self.COLORS['RESET']}"
        record_copy.name = f"\033[94m{record_copy.name}{self.COLORS['RESET']}"  # Blue for logger name
        return super().format(record_copy)


def setup_logging():
    root_logger = logging.getLogger()
    
    if root_logger.handlers:
        return

    error_log_dir = Path("log")
    error_log_dir.mkdir(exist_ok=True)
    error_log_file = error_log_dir / "error.log"

    terminal_level = logging.DEBUG if DEBUG else logging.INFO
    
    terminal_formatter = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-15s:%(lineno)-4d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(terminal_level)
    console_handler.setFormatter(terminal_formatter)
    
    file_handler = logging.handlers.RotatingFileHandler(
        error_log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)

    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def get_logger(name: str | None = None) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name or __name__)