"""Central logging for IDSS. All modules log through get_logger(); never use print or the stdlib logging directly."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger as _logger

from configs.settings import get_settings

if TYPE_CHECKING:
    from loguru import Logger

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"

_configured = False


def setup_logging(level: str | None = None, log_dir: Path | None = None, *, force: bool = False) -> None:
    """Configure console + rotating file sinks. Safe to call repeatedly; only the first call (or force=True) applies."""
    global _configured
    if _configured and not force:
        return

    settings = get_settings()
    level = (level or settings.log_level).upper()
    log_dir = log_dir or settings.logs_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    _logger.remove()
    _logger.add(sys.stderr, level=level, format=_LOG_FORMAT)  # colour only when stderr is a terminal
    _logger.add(
        log_dir / "idss.log",
        level=level,
        format=_FILE_FORMAT,
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
        delay=True,
    )
    _configured = True


def get_logger() -> "Logger":
    """Return the configured loguru logger, setting it up on first use."""
    setup_logging()
    return _logger
