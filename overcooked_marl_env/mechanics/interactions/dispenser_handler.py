"""
Dispenser interaction handlers for Overcooked MARL.
Handles Plate Dispenser ('P') and Ingredient Dispensers ('I' and map dispensers).
"""

from typing import Dict, Tuple, Optional
from ...core.items import (
    create_plate, add_ingredient_to_plate, is_ready
)
from .base_station import StationHandler


class PlateDispenserHandler(StationHandler):
    """Handles interactions at plate dispensers ('P')."""

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
        # Scenario 1: Pick up empty plate
        if not carrying_item:
            plate = create_plate()
            plate["contributors"] = [agent_id]
            plate_instance = self.create_item_instance(plate, game_state)
            game_state["agent_holding_instance"][agent_id] = plate_instance

            # Check if there are ready ingredients on ground nearby (ready to combine)
            ready_ingredient_nearby = self.reward_shaper._check_ready_ingredient_nearby(agent_id, game_state)
            progress_reward = 0.0
            if ready_ingredient_nearby:
                progress_reward = self.rewards.progress_ready_to_combine
                if verbose:
                    print(f"  Ready ingredient nearby - ready to combine!")

            total_reward = self.rewards.pickup + progress_reward
            if total_reward > 0:
                plate["shaping_reward_accumulated"] = plate.get("shaping_reward_accumulated", 0.0) + total_reward
                self.update_item(plate_instance, plate, game_state)

            if verbose:
                print(f"  Picked up an empty plate from dispenser (instance: {plate_instance})" +
                      (f" (+{progress_reward:.1f} ready to combine)" if progress_reward > 0 else ""))
            return total_reward, 0.0

        # Scenario 2: Pick up plate and combine with READY ingredient
        elif carrying_item and is_ready(carrying_item):
            plate = create_plate()
            plate = add_ingredient_to_plate(plate, carrying_item["name"], carrying_item)
            if agent_id not in plate.get("contributors", []):
                plate.setdefault("contributors", []).append(agent_id)
            plate_instance = self.create_item_instance(plate, game_state)

            # Delete the ingredient
            self.delete_item_instance(carrying_instance_id, game_state)
            del game_state["agent_holding_instance"][agent_id]

            # Now carry the plate
            game_state["agent_holding_instance"][agent_id] = plate_instance

            # Calculate progress reward for combining
            progress_reward = self.reward_shaper._calculate_progress_reward(
                'combine', carrying_item, plate, game_state
            )

            pickup_plate_with_ready_reward = self.rewards.progress_pickup_plate_with_ready
            partial_plate_reward = self.reward_shaper._calculate_partial_plate_reward(plate, game_state)

            total_reward = (self.rewards.pickup + self.rewards.combine +
                            progress_reward + pickup_plate_with_ready_reward +
                            partial_plate_reward)

            if total_reward > 0:
                plate["shaping_reward_accumulated"] = plate.get("shaping_reward_accumulated", 0.0) + total_reward
                self.update_item(plate_instance, plate, game_state)

            if verbose:
                print(f"  Picked up plate and combined with {carrying_item['name']} (instance: {plate_instance})" +
                      (f" (+{progress_reward:.1f} progress" if progress_reward > 0 else "") +
                      f" +{pickup_plate_with_ready_reward:.1f} pickup_plate_with_ready" +
                      (f" +{partial_plate_reward:.1f} partial_plate" if partial_plate_reward > 0 else "") + ")")
            return total_reward, 0.0

        # Invalid plate dispenser interaction
        if verbose:
            print(f"  Cannot get plate: carrying non-ready ingredient")
        return self.rewards.invalid, 0.0


class IngredientDispenserHandler(StationHandler):
    """Handles interactions at ingredient dispensers ('I' and map dispensers)."""

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
        # Can only pick up if not carrying anything
        if carrying_item:
            game_state.setdefault("action_history", {}).setdefault(agent_id, []).append(
                ('interact_invalid', tile_pos)
            )
            if len(game_state["action_history"][agent_id]) > 10:
                game_state["action_history"][agent_id].pop(0)

            repetition_penalty = 0.0
            if self.reward_shaper._check_action_repetition(agent_id, 'interact_invalid', game_state):
                repetition_penalty = self.rewards.repetition_penalty

            if verbose:
                print(f"  Cannot pick up ingredient: hands full (penalty: {self.rewards.invalid + repetition_penalty:.2f})")
            return self.rewards.invalid + repetition_penalty, 0.0

        dispenser_data = game_state.get("dispensers", {}).get(tile_pos)
        if not dispenser_data:
            if verbose:
                print(f"  No ingredient configured for dispenser at {tile_pos}")
            return self.rewards.invalid, 0.0

        ingredient_name = dispenser_data if isinstance(dispenser_data, str) else str(dispenser_data)
        ingredient = self.item_manager.create_ingredient_by_name(ingredient_name)
        ingredient["contributors"] = [agent_id]

        # Calculate progress reward before registering instance in hands
        progress_reward = self.reward_shaper._calculate_progress_reward(
            'pickup_ingredient', ingredient, None, game_state
        )

        item_instance = self.create_item_instance(ingredient, game_state)
        game_state["agent_holding_instance"][agent_id] = item_instance
        total_reward = self.rewards.pickup + progress_reward
        if total_reward > 0:
            ingredient["shaping_reward_accumulated"] = (
                ingredient.get("shaping_reward_accumulated", 0.0) + total_reward
            )
            self.update_item(item_instance, ingredient, game_state)

        if verbose:
            print(f"  Picked up {ingredient['name']} from dispenser (instance: {item_instance})" +
                  (f" (+{progress_reward:.1f} progress)" if progress_reward > 0 else ""))
        return total_reward, 0.0
