"""
Delivery station interaction handler for Overcooked MARL.
Handles recipe validation, HUD queue order matching, dynamic multipliers,
and episode victory bonuses ('D').
"""

from typing import Dict, Tuple, Optional
from ...core.items import ItemType, is_plate
from ...core.recipes import OrderQueue
from .base_station import StationHandler


class DeliveryHandler(StationHandler):
    """Handles interactions at delivery stations ('D')."""

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
        invalid_pen = self.rewards.invalid_delivery_penalty

        # Must be carrying a plate
        if not carrying_item or (carrying_item.get("type") != ItemType.DISH and not is_plate(carrying_item)):
            if verbose:
                print(f"  Cannot deliver: not carrying a plate")
            return invalid_pen, 0.0

        # Plate must have ingredients
        ingredients = carrying_item.get("ingredients", [])
        if not ingredients:
            if verbose:
                print(f"  Cannot deliver: plate is empty")
            return invalid_pen, 0.0

        # Validate recipe
        is_valid, recipe_name, recipe_reward = self.recipe_manager.validate_delivery(ingredients)

        if not is_valid:
            if verbose:
                ingredients_str = ", ".join(ingredients)
                print(f"  Invalid recipe! Ingredients: [{ingredients_str}] (penalty: {invalid_pen})")
            return invalid_pen, 0.0

        # Get remaining pending orders in HUD queue
        pending_orders = OrderQueue.get_pending(
            game_state.get("recipe_order", []),
            game_state.get("delivered_recipes", [])
        )

        # Must match a recipe currently requested in pending HUD orders
        if recipe_name not in pending_orders:
            unrequested_penalty = -abs(recipe_reward)
            if verbose:
                print(f"  Recipe {recipe_name} is not in pending HUD orders! Wiping dish points (penalty: {unrequested_penalty:.1f})")

            # Delete unrequested plate to clear hands, but do NOT count as successful delivery
            self.delete_item_instance(carrying_instance_id, game_state)
            del game_state["agent_holding_instance"][agent_id]
            return unrequested_penalty, unrequested_penalty

        # Check if it's strictly the top order in HUD queue (for order bonus)
        is_correct_order = (len(pending_orders) > 0 and recipe_name == pending_orders[0])

        # Register delivery
        game_state["delivered_recipes"].append(recipe_name)
        game_state["dishes_pending"] -= 1
        if "expected_recipe_index" in game_state:
            game_state["expected_recipe_index"] += 1

        # Delete delivered plate
        self.delete_item_instance(carrying_instance_id, game_state)
        del game_state["agent_holding_instance"][agent_id]

        # Calculate rewards
        base_reward = recipe_reward
        order_bonus = (recipe_reward * self.rewards.order_bonus_mult) if is_correct_order else 0.0

        # Sequence complete bonus
        sequence_bonus = self.rewards.sequence_complete_bonus

        # Dynamic delivery multipliers for any N dishes (1.0 for 1st, +0.5 per extra delivery)
        num_deliveries = len(game_state.get("delivered_recipes", []))
        mult_step = getattr(self.rewards, "delivery_multiplier_step", 0.5)
        delivery_multiplier = 1.0 + max(0, num_deliveries - 1) * mult_step

        delivery_bonus = base_reward * (delivery_multiplier - 1.0)

        # Check if episode is now complete (all dishes delivered)
        episode_complete_bonus = 0.0
        if game_state["dishes_pending"] <= 0:
            episode_complete_bonus = self.rewards.episode_complete_bonus
            if verbose:
                print(f"  ALL DISHES DELIVERED! Episode Victory Bonus: +{episode_complete_bonus}")

        total_reward = base_reward + order_bonus + sequence_bonus + delivery_bonus + episode_complete_bonus
        global_reward_bonus = recipe_reward + order_bonus + delivery_bonus + episode_complete_bonus

        if verbose:
            if is_correct_order:
                print(f"  Recipe {recipe_name} delivered in correct order! (Delivery #{num_deliveries})")
                print(f"  Reward: {base_reward} + Order Bonus: {order_bonus} + Sequence Bonus: {sequence_bonus} + Delivery Bonus: {delivery_bonus:.1f} (x{delivery_multiplier:.1f})" +
                      (f" + Victory Bonus: {episode_complete_bonus}" if episode_complete_bonus > 0 else ""))
            else:
                print(f"  Recipe {recipe_name} delivered out of order (Delivery #{num_deliveries})")
                print(f"  Reward: {base_reward} + Sequence Bonus: {sequence_bonus} + Delivery Bonus: {delivery_bonus:.1f} (x{delivery_multiplier:.1f}) (no order bonus)" +
                      (f" + Victory Bonus: {episode_complete_bonus}" if episode_complete_bonus > 0 else ""))

        return total_reward, global_reward_bonus
