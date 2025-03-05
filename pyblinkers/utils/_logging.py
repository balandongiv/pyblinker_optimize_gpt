import logging
import warnings
import inspect

logger = logging.getLogger("pyblinkers")
logger.propagate = False  # Prevent duplicate logs

class _FrameFilter(logging.Filter):
    """Filter to add frame information to log messages."""
    def __init__(self):
        self.add_frames = 0

    def filter(self, record):
        record.frame_info = "Unknown"
        if self.add_frames:
            frame_info = _frame_info(5 + self.add_frames)[5:][::-1]
            if frame_info:
                frame_info[-1] = (frame_info[-1] + " :").ljust(30)
                if len(frame_info) > 1:
                    frame_info[0] = "┌" + frame_info[0]
                    frame_info[-1] = "└" + frame_info[-1]
                for i in range(1, len(frame_info) - 1):
                    frame_info[i] = "├" + frame_info[i]
                record.frame_info = "\n".join(frame_info)
        return True

_filter = _FrameFilter()
logger.addFilter(_filter)

def set_log_level(level="INFO"):
    """Set the logging level."""
    levels = {"DEBUG": logging.DEBUG, "INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}
    logger.setLevel(levels.get(level.upper(), logging.INFO))

def set_log_file(filename=None):
    """Redirect logs to a file."""
    logger.handlers.clear()
    handler = logging.FileHandler(filename) if filename else logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(frame_info)s %(message)s"))
    logger.addHandler(handler)

def warn(message, category=RuntimeWarning, verbose=True):
    """Emit a warning and log it based on verbosity."""
    if verbose:
        warnings.warn(message, category)
        logger.warning(message)

def _frame_info(n):
    """Retrieve frame information for logging."""
    frame = inspect.currentframe()
    try:
        frame = frame.f_back
        infos = []
        for _ in range(n):
            try:
                name = frame.f_globals["__name__"]
            except KeyError:
                pass
            else:
                infos.append(f"{name}:{frame.f_lineno}")
            frame = frame.f_back
            if frame is None:
                break
        return infos
    except Exception:
        return ["unknown"]
    finally:
        del frame

