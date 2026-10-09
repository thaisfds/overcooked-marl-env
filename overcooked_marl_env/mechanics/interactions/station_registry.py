"""
Station Registry for Overcooked MARL.
Allows dynamic registration and lookup of StationHandlers by station key.
"""

from typing import Dict, Optional
from .base_station import StationHandler


class StationRegistry:
    """Registry managing StationHandlers for various stations in the environment."""

    def __init__(self):
        self._handlers: Dict[str, StationHandler] = {}

    def register(self, station_key: str, handler: StationHandler):
        """Register a handler for a station key (e.g. 'C', 'P', 'D', 'T', '#', 'I')."""
        self._handlers[station_key] = handler

    def get(self, station_key: str) -> Optional[StationHandler]:
        """Retrieve handler for station key, or None if not found."""
        return self._handlers.get(station_key)

    def __contains__(self, station_key: str) -> bool:
        return station_key in self._handlers
