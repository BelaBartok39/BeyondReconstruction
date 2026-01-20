"""TorchRF Testbed - Live RF anomaly detection with HackRF."""

# Lazy imports to avoid circular dependency issues with CLP_Project

__all__ = [
    "LiveDetector",
    "SessionRecorder",
    "inject_anomaly",
    "normalize_signal",
    "estimate_snr",
    "estimate_power",
]


def __getattr__(name):
    """Lazy import attributes on first access."""
    if name == "LiveDetector":
        from .detector import LiveDetector
        return LiveDetector
    elif name == "SessionRecorder":
        from .recorder import SessionRecorder
        return SessionRecorder
    elif name == "inject_anomaly":
        from .injection import inject_anomaly
        return inject_anomaly
    elif name in ("normalize_signal", "estimate_snr", "estimate_power"):
        from . import utils
        return getattr(utils, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
