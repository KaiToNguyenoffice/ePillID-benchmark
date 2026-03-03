"""
Azure ML abstraction layer.

Provides a unified interface that works both with and without Azure ML SDK.
When the SDK is not installed, all operations become no-ops so the training
pipeline can run locally without modification.
"""

import logging

logger = logging.getLogger(__name__)


class _LocalRun:
    """Drop-in replacement for azureml.core.run.Run when SDK is unavailable."""

    def __init__(self):
        self._tags = {}
        self._metrics = {}

    def tag(self, key, value):
        self._tags[key] = value

    def log(self, name, value):
        self._metrics[name] = value
        logger.info("metric  %s = %s", name, value)

    def log_image(self, name, plot=None, path=None, description=None):
        logger.info("image   %s (plot object logged locally)", name)

    def get_portal_url(self):
        return "local://run"


def get_run():
    """Return an Azure ML Run context or a local stub."""
    try:
        from azureml.core.run import Run
        run = Run.get_context()
        logger.info("Using Azure ML Run context")
        return run
    except (ImportError, ModuleNotFoundError):
        logger.info("Azure ML SDK not found – using local run stub")
        return _LocalRun()
    except Exception as exc:
        logger.warning("Azure ML init failed (%s) – falling back to local stub", exc)
        return _LocalRun()
