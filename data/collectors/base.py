"""
Base collector class. All collectors inherit from this.
Enforces a consistent interface: collect() returns a dict, table_name identifies storage target.
"""
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone


class BaseCollector(ABC):
    """Abstract base for all data collectors."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    @abstractmethod
    def table_name(self) -> str:
        """Which store table this collector writes to."""
        pass

    @abstractmethod
    def collect(self) -> dict | None:
        """
        Fetch data and return a dict of values.
        Returns None if collection fails (logged internally).
        The dict should NOT include 'timestamp' — that's added by the runner.
        """
        pass

    def collect_with_timestamp(self) -> dict | None:
        """Wrapper that adds UTC timestamp and handles errors."""
        try:
            data = self.collect()
            if data is None:
                self.logger.warning("Collection returned None")
                return None
            # Round to nearest 4-hour boundary for alignment
            now = datetime.now(timezone.utc)
            hour = (now.hour // 4) * 4
            aligned = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            data["timestamp"] = aligned.isoformat()
            return data
        except Exception as e:
            self.logger.error(f"Collection failed: {e}", exc_info=True)
            return None
