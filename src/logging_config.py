"""
One place that sets up logging for the whole service. Called once at
startup (app/main.py does this), then every other module just does
logging.getLogger(__name__) and uses it, no print() anywhere in src/ or app/.
"""

import logging
import logging.handlers

from src.config import PROJECT_ROOT, load_config

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

_already_configured = False


def setup_logging() -> None:
    global _already_configured
    if _already_configured:
        return

    config = load_config()["logging"]

    log_dir = PROJECT_ROOT / config["dir"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / config["file"]

    level = getattr(logging, config["level"].upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    formatter = logging.Formatter(_LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=config.get("max_bytes", 5_000_000),
        backupCount=config.get("backup_count", 3),
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _already_configured = True
