"""
Centralised logging configuration.
Creates a file handler (logs/anc_runtime.log) plus a console handler.

Windows note: the Windows console defaults to cp1252, which cannot encode
many Unicode characters.  The console stream is opened with errors='replace'
so any unencodable character is substituted with '?' instead of crashing.
"""
import io
import logging
import sys
from pathlib import Path

LOG_DIR  = Path(__file__).resolve().parent.parent / "logs"
LOG_FILE = LOG_DIR / "anc_runtime.log"

_FMT      = "%(asctime)s [%(levelname)-8s] %(name)-20s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"


def _safe_console_stream():
    """
    Return a stdout stream that never raises UnicodeEncodeError.

    On Windows the default stdout uses the system code-page (e.g. cp1252).
    We re-wrap the underlying binary buffer with UTF-8 + errors='replace'
    so log messages containing arrows, ellipses, or other non-ASCII chars
    are printed as '?' rather than crashing the process.
    """
    try:
        # sys.stdout.buffer is the raw binary stream (not available in IDLE/some IDEs)
        return io.TextIOWrapper(
            sys.stdout.buffer,
            encoding="utf-8",
            errors="replace",
            line_buffering=True,
        )
    except AttributeError:
        # Fallback: plain sys.stdout (already text; some envs don't expose .buffer)
        return sys.stdout


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Return a named logger.  Safe to call multiple times for the same name —
    handlers are only attached once.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    # Console — INFO and above, Unicode-safe on Windows
    console = logging.StreamHandler(_safe_console_stream())
    console.setFormatter(formatter)
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    # File — DEBUG and above (persistent for post-mortem analysis)
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
    except OSError as exc:
        logger.warning("Could not open log file '%s': %s", LOG_FILE, exc)

    return logger
