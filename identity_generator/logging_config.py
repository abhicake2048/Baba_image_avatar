from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


LOGGER_NAME = "identity_generator"
LOG_FILENAME = "app.log"


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def configure_logging(log_dir: Path, level: str = "INFO") -> Path:
    directory = log_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    log_path = directory / LOG_FILENAME

    logger = get_logger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    for handler in list(logger.handlers):
        if getattr(handler, "_portrait_studio_handler", False):
            handler.close()
            logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler._portrait_studio_handler = True  # type: ignore[attr-defined]

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler._portrait_studio_handler = True  # type: ignore[attr-defined]

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.info("logger.ready path=%s level=%s", log_path, logging.getLevelName(logger.level))
    return log_path


def tail_log(log_path: Path, max_lines: int = 200) -> str:
    if not log_path.is_file():
        return "The log file has not been created yet."
    try:
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    except OSError as exc:
        return f"Could not read the log file: {exc}"
    return "".join(lines[-max_lines:]).rstrip() or "The log file is empty."

