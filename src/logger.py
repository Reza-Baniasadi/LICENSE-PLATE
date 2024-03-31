import logging
from pathlib import Path


def initialize_logger(logger_name: str, file_output: str = None):
    log = logging.getLogger(logger_name)
    log.setLevel(logging.INFO)

    if log.hasHandlers():
        log.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )
    console_handler.setFormatter(console_format)
    log.addHandler(console_handler)

    if file_output:
        log_file_path = Path(file_output)
        log_file_path.parent.mkdir(parents=True, exist_ok=True) 
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(console_format)
        log.addHandler(file_handler)

    return log
