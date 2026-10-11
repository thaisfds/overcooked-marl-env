"""
Chopping board interaction handler.
Handles cutting ingredients, placing raw ingredients, picking up processed ingredients,
and direct assembly onto a held plate.
"""

from typing import Dict, Tuple, Optional, Any
from ...core.items import (
    ItemType, can_process, is_ready, is_plate,
    advance_processing, add_ingredient_to_plate
)
from .base_station import StationHandler


class ChoppingBoardHandler(StationHandler):
    """Handles interactions at processing stations ('C' by default, or custom appliance)."""

    def __init__(self, context: Any, station_char: str = 'C'):
        super().__init__(context)
        self.station_char = station_char

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
        ix, iy = tile_pos

        # Scenario 1: Place ingredient on chopping board / station
        if carrying_item and can_process(carrying_item, station=self.station_char) and not ground_item:
            if agent_id not in carrying_item.get("contributors", []):
                carrying_item.setdefault("contributors", []).append(agent_id)
            carrying_item["last_placed_by"] = agent_id

            game_state["ground_instances"][(ix, iy)] = carrying_instance_id
            del game_state["agent_holding_instance"][agent_id]

            # Calculate progress reward
            place_action = 'place_on_chop' if self.station_char == 'C' else 'place_on_station'
            progress_reward = self.reward_shaper._calculate_progress_reward(
                place_action, carrying_item, None, game_state
            )
            total_reward = self.rewards.place_on_chop + progress_reward
            if total_reward > 0:
                carrying_item["shaping_reward_accumulated"] = (
                    carrying_item.get("shaping_reward_accumulated", 0.0) + total_reward
                )

            if verbose:
                station_name = "chopping board" if self.station_char == 'C' else f"station '{self.station_char}'"
                print(f"  Placed {carrying_item['name']} on {station_name} (+{self.rewards.place_on_chop}" +
                      (f" +{progress_reward:.1f} progress" if progress_reward > 0 else "") + ")")
            return total_reward, 0.0

        # Scenario 2: Chop/process ingredient on station
        elif not carrying_item and ground_item and can_process(ground_item, station=self.station_char):
            # Register action in history for repetition detection
            game_state.setdefault("action_history", {}).setdefault(agent_id, []).append(
                ('process', tile_pos)
            )
            # Limit history to 10 actions
            if len(game_state["action_history"][agent_id]) > 10:
                game_state["action_history"][agent_id].pop(0)

            updated_item = advance_processing(ground_item, station=self.station_char)
            if agent_id not in updated_item.get("contributors", []):
                updated_item.setdefault("contributors", []).append(agent_id)

            # Check for repetition penalty
            repetition_penalty = 0.0
            if self.reward_shaper._check_action_repetition(agent_id, 'process', game_state):
                repetition_penalty = self.rewards.repetition_penalty
                if verbose:
                    print(f"  WARNING: Repetitive processing detected! Penalty: {repetition_penalty}")

            # Check for accumulation penalty (many processed ingredients on ground)
            accumulation_penalty = 0.0
            processed_count = self.reward_shaper._count_processed_ingredients_on_ground(game_state)
            if processed_count >= self.rewards.accumulation_threshold:
                excess = processed_count - self.rewards.accumulation_threshold + 1
                accumulation_penalty = self.rewards.accumulation_penalty * excess
                if verbose:
                    print(f"  WARNING: Too many processed ingredients on ground ({processed_count})! Penalty: {accumulation_penalty}")

            cut_reward = self.rewards.process if getattr(self.rewards, "process", 1.0) != 1.0 else self.rewards.chop_cut
            is_stage_completed = is_ready(updated_item) or not can_process(updated_item, station=self.station_char)
            milestone_reward = self.rewards.chop_completed if is_stage_completed else 0.0
            total_reward = cut_reward + milestone_reward + repetition_penalty + accumulation_penalty

            reward_gained = max(0.0, cut_reward + milestone_reward)
            updated_item["shaping_reward_accumulated"] = (
                updated_item.get("shaping_reward_accumulated", 0.0) + reward_gained
            )
            self.update_item(ground_instance_id, updated_item, game_state)

            if verbose:
                cuts_done = updated_item.get("cuts_completed", updated_item.get("cuts_done", 0))
                cuts_needed = updated_item.get("cuts_needed", 3)
                print(f"  Cutting... {cuts_done}/{cuts_needed} cuts (+{cut_reward:.1f}" +
                      (f" +{milestone_reward:.1f} complete milestone" if milestone_reward > 0 else "") + ")")
                if is_ready(updated_item):
                    print(f"  Item processed and ready to use!")
                elif is_stage_completed:
                    print(f"  Item stage completed on {self.station_char}!")

            return total_reward, 0.0

        # Scenario 3: Pick up processed ingredient from chopping board / station
        elif not carrying_item and ground_item and (is_ready(ground_item) or not can_process(ground_item, station=self.station_char)):
            if agent_id not in ground_item.get("contributors", []):
                ground_item.setdefault("contributors", []).append(agent_id)
            ground_item["last_placed_by"] = None

            progress_reward = 0.0
            if is_ready(ground_item):
                progress_reward = self.reward_shaper._calculate_progress_reward(
                    'pickup_ready', ground_item, None, game_state
                )

            game_state["agent_holding_instance"][agent_id] = ground_instance_id
            del game_state["ground_instances"][(ix, iy)]

            total_reward = self.rewards.pickup + progress_reward
            if total_reward > 0:
                ground_item["shaping_reward_accumulated"] = (
                    ground_item.get("shaping_reward_accumulated", 0.0) + total_reward
                )

            if verbose:
                station_name = "chopping board" if self.station_char == 'C' else f"station '{self.station_char}'"
                print(f"  Picked up {ground_item['name']} from {station_name} (+{self.rewards.pickup}" +
                      (f" +{progress_reward:.1f} progress" if progress_reward > 0 else "") + ")")
            return total_reward, 0.0

        # Scenario 4: Carrying plate + ready ingredient on chopping board -> combine directly onto plate!
        elif carrying_item and (carrying_item.get("type") == ItemType.DISH or is_plate(carrying_item)) and ground_item and is_ready(ground_item):
            if ground_item["name"] in carrying_item.get("ingredients", []):
                if verbose:
                    print(f"  Cannot combine: {ground_item['name']} already present on plate")
                return self.rewards.invalid, 0.0

            updated_plate = add_ingredient_to_plate(carrying_item, ground_item["name"], ground_item)
            if agent_id not in updated_plate.get("contributors", []):
                updated_plate.setdefault("contributors", []).append(agent_id)

            # Delete ingredient from chopping board
            self.delete_item_instance(ground_instance_id, game_state)
            del game_state["ground_instances"][(ix, iy)]

            progress_reward = self.reward_shaper._calculate_progress_reward(
                'combine', carrying_item, ground_item, game_state
            )
            partial_plate_reward = self.reward_shaper._calculate_partial_plate_reward(updated_plate, game_state)

            total_reward = self.rewards.combine + progress_reward + partial_plate_reward
            if total_reward > 0:
                updated_plate["shaping_reward_accumulated"] = (
                    updated_plate.get("shaping_reward_accumulated", 0.0) + total_reward
                )
            self.update_item(carrying_instance_id, updated_plate, game_state)

            if verbose:
                print(f"  Combined {ground_item['name']} from chopping board with plate (+{self.rewards.combine}" +
                      (f" +{progress_reward:.1f} progress" if progress_reward > 0 else "") +
                      (f" +{partial_plate_reward:.1f} partial_plate" if partial_plate_reward > 0 else "") + ")")
            return total_reward, 0.0

        # Invalid chopping board interaction
        game_state.setdefault("action_history", {}).setdefault(agent_id, []).append(
            ('interact_invalid', tile_pos)
        )
        if len(game_state["action_history"][agent_id]) > 10:
            game_state["action_history"][agent_id].pop(0)

        repetition_penalty = 0.0
        if self.reward_shaper._check_action_repetition(agent_id, 'interact_invalid', game_state):
            repetition_penalty = self.rewards.repetition_penalty

        if verbose:
            print(f"  Invalid chopping board interaction (penalty: {self.rewards.invalid + repetition_penalty:.2f})")
        return self.rewards.invalid + repetition_penalty, 0.0
