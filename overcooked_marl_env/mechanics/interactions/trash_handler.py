"""
Trash station interaction handler for Overcooked MARL.
Handles deleting items with shaping reward clawback ('T').
"""

from typing import Dict, Tuple, Optional
from ...core.items import get_ingredient_display_name
from .base_station import StationHandler


class TrashHandler(StationHandler):
    """Handles interactions at trash station ('T')."""

    def handle(
        self,
        agent_id: str,
        tile_pos: Tuple[int, int],
        carrying_item: Optional[Dict],
        carrying_instance_id: Optional[int],
        ground_item: Optional[Dict],
        ground_instance_id: Optional[int],
        game_state: Dict,
        verbose: bool = False
    ) -> Tuple[float, float]:
        if carrying_item:
            shaping_accumulated = carrying_item.get("shaping_reward_accumulated", 0.0)
            penalty = -shaping_accumulated + self.rewards.trash_penalty

            self.delete_item_instance(carrying_instance_id, game_state)
            del game_state["agent_holding_instance"][agent_id]

            if verbose:
                print(f"  Discarded {get_ingredient_display_name(carrying_item)} in trash (Estorno: -{shaping_accumulated:.1f}, Taxa: {self.rewards.trash_penalty:.1f} => Total: {penalty:.1f})")
            return penalty, 0.0

        if verbose:
            print(f"  Cannot use trash: not carrying anything")
        return 0.0, 0.0
