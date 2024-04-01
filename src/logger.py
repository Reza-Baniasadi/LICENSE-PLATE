import logging
from pathlib import Path

def create_logger(name: str, save_path: str = None):
    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(logging.INFO)

    if logger_instance.hasHandlers():
        logger_instance.handlers.clear()

    stream_hdlr = logging.StreamHandler()
    stream_hdlr.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s >> %(name)s >> %(levelname)s >> %(message)s")
    stream_hdlr.setFormatter(formatter)
    logger_instance.addHandler(stream_hdlr)

    if save_path:
        path_obj = Path(save_path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        file_hdlr = logging.FileHandler(path_obj)
        file_hdlr.setLevel(logging.INFO)
        file_hdlr.setFormatter(formatter)
        logger_instance.addHandler(file_hdlr)

    return logger_instance
