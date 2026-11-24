"""
Counter interaction handler for Overcooked MARL.
Handles placing, picking up, and combining items on standard counters ('#').
"""

from typing import Dict, Tuple, Optional
from ...core.items import (
    ItemType, is_ready, is_plate, add_ingredient_to_plate, get_ingredient_display_name
)
from ...core.recipes import OrderQueue
from .base_station import StationHandler


class CounterHandler(StationHandler):
    """Handles interactions at counters ('#')."""

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

        is_carrying_dish = is_plate(carrying_item)
        is_carrying_ready = carrying_item is not None and is_ready(carrying_item)
        is_ground_dish = is_plate(ground_item)
        is_ground_ready = ground_item is not None and is_ready(ground_item)

        # Detect useless cycle (picking up/placing repeatedly in the same spot)
        is_useless_cycle = False
        history = game_state.get("action_history", {}).get(agent_id, [])

        if len(history) >= 3:
            recent = history[-3:]
            actions = [a[0] for a in recent]
            positions = [a[1] for a in recent]

            is_alternating = all(actions[i] != actions[i + 1] for i in range(len(actions) - 1))
            same_pos = len(set(positions)) == 1 and positions[0] == tile_pos
            only_pickup_place = all(a in ["pickup", "place"] for a in actions)

            is_useless_cycle = is_alternating and same_pos and only_pickup_place

        # Record action in history
        action_type = "place" if carrying_item else "pickup"
        game_state.setdefault("action_history", {}).setdefault(agent_id, []).append(
            (action_type, tile_pos)
        )
        if len(game_state["action_history"][agent_id]) > 10:
            game_state["action_history"][agent_id].pop(0)

        # Scenario 1: Carrying READY item + plate on counter -> combine
        if is_carrying_ready and is_ground_dish:
            if carrying_item["name"] in ground_item.get("ingredients", []):
                if verbose:
                    print(f"  Cannot combine: {carrying_item['name']} already present on plate")
                return self.rewards.invalid, 0.0

            updated_plate = add_ingredient_to_plate(ground_item, carrying_item["name"], carrying_item)
            if agent_id not in updated_plate.get("contributors", []):
                updated_plate.setdefault("contributors", []).append(agent_id)

            self.delete_item_instance(carrying_instance_id, game_state)
            del game_state["agent_holding_instance"][agent_id]

            progress_reward = self.reward_shaper._calculate_progress_reward(
                'combine', carrying_item, ground_item, game_state
            )
            partial_plate_reward = self.reward_shaper._calculate_partial_plate_reward(updated_plate, game_state)

            total_reward = self.rewards.combine + progress_reward + partial_plate_reward
            if total_reward > 0:
                updated_plate["shaping_reward_accumulated"] = (
                    updated_plate.get("shaping_reward_accumulated", 0.0) + total_reward
                )

            self.update_item(ground_instance_id, updated_plate, game_state)

            if verbose:
                print(f"  Combined {carrying_item['name']} with plate on counter (+{self.rewards.combine}" +
                      (f" +{progress_reward:.1f} progress" if progress_reward > 0 else "") +
                      (f" +{partial_plate_reward:.1f} partial_plate" if partial_plate_reward > 0 else "") + ")")
            return total_reward, 0.0

        # Scenario 2: Carrying plate + READY item on counter -> pick up and combine
        elif is_carrying_dish and is_ground_ready:
            if ground_item["name"] in carrying_item.get("ingredients", []):
                if verbose:
                    print(f"  Cannot combine: {ground_item['name']} already present on plate")
                return self.rewards.invalid, 0.0

            updated_plate = add_ingredient_to_plate(carrying_item, ground_item["name"], ground_item)
            if agent_id not in updated_plate.get("contributors", []):
                updated_plate.setdefault("contributors", []).append(agent_id)

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
                print(f"  Combined {ground_item['name']} from counter with plate (+{self.rewards.combine}" +
                      (f" +{progress_reward:.1f} progress" if progress_reward > 0 else "") +
                      (f" +{partial_plate_reward:.1f} partial_plate" if partial_plate_reward > 0 else "") + ")")
            return total_reward, 0.0

        # Scenario 3: Place item on counter (if empty)
        elif carrying_item and not ground_item:
            # Tag item with last_placed_by and contributor
            carrying_item["last_placed_by"] = agent_id
            if agent_id not in carrying_item.get("contributors", []):
                carrying_item.setdefault("contributors", []).append(agent_id)

            game_state["ground_instances"][(ix, iy)] = carrying_instance_id
            del game_state["agent_holding_instance"][agent_id]

            counter_code = self.item_manager.get_cell_code('#')
            total_counter_items = sum(
                1 for pos, _ in game_state["ground_instances"].items()
                if game_state["map_static"][pos[1], pos[0]] == counter_code
            )

            reward = self.rewards.place
            if not is_useless_cycle:
                # Stage identification to allow legitimate handoff placement once per item stage
                # without enabling infinite place/pickup reward farming loops
                if is_carrying_dish:
                    place_stage = ("plate", tuple(sorted(carrying_item.get("ingredients", []))))
                else:
                    place_stage = (
                        carrying_item.get("name"),
                        carrying_item.get("state"),
                        carrying_item.get("cuts_completed", carrying_item.get("cuts_done", 0))
                    )
                placed_stages = carrying_item.setdefault("place_stages_awarded", set())
                if place_stage not in placed_stages:
                    placed_stages.add(place_stage)
                    reward += getattr(self.rewards, "handoff_place", 1.0)
                    self.update_item(carrying_instance_id, carrying_item, game_state)
            else:
                reward += self.rewards.useless_cycle_penalty
                if verbose:
                    print(f"  WARNING: Useless pickup/place cycle detected! Penalty: {self.rewards.useless_cycle_penalty}")

            # Penalize abandoning a completed dish that matches a pending recipe
            if is_carrying_dish and carrying_item.get("ingredients"):
                is_valid, recipe_name, _ = self.recipe_manager.validate_delivery(carrying_item.get("ingredients", []))
                pending_recipes = OrderQueue.get_pending(
                    game_state.get("recipe_order", []),
                    game_state.get("delivered_recipes", [])
                )
                if is_valid and recipe_name in pending_recipes:
                    abandon_pen = getattr(self.rewards, "abandon_dish_penalty", -10.0)
                    reward += abandon_pen
                    if verbose:
                        print(f"  WARNING: Abandoned completed dish '{recipe_name}' on counter! Penalty: {abandon_pen}")

            if total_counter_items > 3:
                excess_items = total_counter_items - 3
                clutter_penalty = self.rewards.counter_clutter_penalty * (excess_items ** 1.5)
                reward += clutter_penalty
                if verbose:
                    print(f"  Placed {get_ingredient_display_name(carrying_item)} on counter")
                    print(f"  Counter clutter penalty: {clutter_penalty:.2f} ({total_counter_items} items on counters)")
            else:
                if verbose:
                    print(f"  Placed {get_ingredient_display_name(carrying_item)} on counter")

            return reward, 0.0

        # Scenario 4: Pick up item from counter (if hands empty)
        elif not carrying_item and ground_item:
            game_state["agent_holding_instance"][agent_id] = ground_instance_id
            del game_state["ground_instances"][(ix, iy)]

            progress_reward = 0.0
            handoff_local_reward = 0.0
            handoff_global_reward = 0.0

            # Cooperative handoff detection:
            # If item was placed by another agent, award cooperative handoff bonus!
            last_placed_by = ground_item.get("last_placed_by")
            if last_placed_by is not None and last_placed_by != agent_id:
                # Stage identification to avoid back-and-forth handoff farming
                if is_plate(ground_item):
                    current_stage = ("plate", tuple(sorted(ground_item.get("ingredients", []))))
                else:
                    current_stage = (
                        ground_item.get("name"),
                        ground_item.get("state"),
                        ground_item.get("cuts_completed", ground_item.get("cuts_done", 0))
                    )

                awarded_stages = ground_item.setdefault("handoff_stages_awarded", set())
                if current_stage not in awarded_stages:
                    awarded_stages.add(current_stage)
                    coop_val = getattr(self.rewards, "cooperative_handoff", 5.0)
                    # Split reward: local bonus for the picker, global bonus so the placer benefits too!
                    handoff_local_reward = coop_val / 2.0
                    handoff_global_reward = coop_val / 2.0
                    if verbose:
                        print(f"  COOPERATIVE HANDOFF! {agent_id} picked up item placed by {last_placed_by} (+{handoff_local_reward:.1f} local, +{handoff_global_reward:.1f} global)")

            # Record this agent as contributor and clear last_placed_by
            if agent_id not in ground_item.get("contributors", []):
                ground_item.setdefault("contributors", []).append(agent_id)
            ground_item["last_placed_by"] = None

            if is_ready(ground_item) and not ground_item.get("pickup_ready_awarded", False):
                progress_reward = self.reward_shaper._calculate_progress_reward(
                    'pickup_ready', ground_item, None, game_state
                )
                if progress_reward > 0:
                    ground_item["pickup_ready_awarded"] = True
                    self.update_item(ground_instance_id, ground_item, game_state)
            elif is_plate(ground_item):
                current_combo = tuple(sorted(ground_item.get("ingredients", [])))
                last_awarded_combo = ground_item.get("pickup_awarded_combo", None)
                if current_combo != last_awarded_combo and len(current_combo) > 0:
                    progress_reward = self.reward_shaper._calculate_progress_reward(
                        'pickup_plate_complete', ground_item, None, game_state
                    )
                    if progress_reward > 0:
                        ground_item["pickup_awarded_combo"] = current_combo

            total_reward = self.rewards.pickup + progress_reward + handoff_local_reward
            if total_reward > 0:
                ground_item["shaping_reward_accumulated"] = (
                    ground_item.get("shaping_reward_accumulated", 0.0) + total_reward
                )
                self.update_item(ground_instance_id, ground_item, game_state)

            if verbose:
                print(f"  Picked up {get_ingredient_display_name(ground_item)} from counter (+{self.rewards.pickup}" +
                      (f" +{progress_reward:.1f} progress" if progress_reward > 0 else "") +
                      (f" +{handoff_local_reward:.1f} handoff" if handoff_local_reward > 0 else "") + ")")

            return total_reward, handoff_global_reward

        # Invalid counter interaction
        if verbose:
            print(f"  Invalid counter interaction (penalty: {self.rewards.invalid})")
        return self.rewards.invalid, 0.0
