"""
Interaction handling module for Overcooked MARL.

This module implements the interaction system using the Strategy pattern,
allowing for extensible and maintainable interaction logic.
"""

from .base_station import StationHandler
from .station_registry import StationRegistry
from .chopping_handler import ChoppingBoardHandler
from .dispenser_handler import PlateDispenserHandler, IngredientDispenserHandler
from .counter_handler import CounterHandler
from .trash_handler import TrashHandler
from .delivery_handler import DeliveryHandler

__all__ = [
    "StationHandler",
    "StationRegistry",
    "ChoppingBoardHandler",
    "PlateDispenserHandler",
    "IngredientDispenserHandler",
    "CounterHandler",
    "TrashHandler",
    "DeliveryHandler",
]
