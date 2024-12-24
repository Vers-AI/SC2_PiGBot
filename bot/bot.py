# bot.py

from typing import Optional
from itertools import cycle, chain
import numpy as np

# python-sc2 & ares imports (adjust as necessary)
from sc2.data import Result
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units

from ares import AresBot
from ares.consts import ALL_STRUCTURES, WORKER_TYPES, UnitRole
from ares.managers.manager_mediator import ManagerMediator

# Cython or custom extension references (if you need them)
from cython_extensions import (
    cy_closest_to, cy_pick_enemy_target, 
    cy_find_units_center_mass, cy_attack_ready, 
    cy_unit_pending, cy_distance_to
)

# -------------
# Our new modules
# -------------
# Macro
from bot.hub.macro import handle_macro, worker_production

# Combat
from bot.hub.combat import (
    threat_detection,
    control_main_army,    # If you still use it directly
    warp_prism_follower,  # If you still use it directly
)

# Scouting
from bot.hub.scouting import control_scout

# Reactions
from bot.hub.reactions import (
    early_threat_sensor,
    cheese_reaction,
    defend_worker_cannon_rush,
    one_base_reaction
)


COMMON_UNIT_IGNORE_TYPES: set[UnitTypeId] = {
    UnitTypeId.EGG,
    UnitTypeId.LARVA,
    UnitTypeId.CREEPTUMORBURROWED,
    UnitTypeId.CREEPTUMORQUEEN,
    UnitTypeId.CREEPTUMOR,
    UnitTypeId.MULE,
    UnitTypeId.PROBE,
    UnitTypeId.SCV,
    UnitTypeId.DRONE,
    UnitTypeId.OVERLORD,
    UnitTypeId.OVERSEER,
    UnitTypeId.LOCUSTMP,
    UnitTypeId.LOCUSTMPFLYING,
    UnitTypeId.ADEPTPHASESHIFT,
    UnitTypeId.CHANGELING,
    UnitTypeId.CHANGELINGMARINE,
    UnitTypeId.CHANGELINGZEALOT,
    UnitTypeId.CHANGELINGZERGLING,
}


class DragonBot(AresBot):
    current_base_target: Point2
    expansions_generator: cycle
    _begin_attack_at_supply: float

    def __init__(self, game_step_override: Optional[int] = None):
        super().__init__(game_step_override)

        # State variables
        self.unit_roles = {}
        self.scout_targets = {}
        self.bases = {}
        self.total_health_shield_percentage = 1.0

        self._commenced_attack = False
        self._used_cheese_response = False
        self._used_one_base_response = False
        self._under_attack = False
        self._cheese_reaction_completed = False
        self._one_base_reaction_completed = False
        self._is_building = False

        self.resource_by_tag = {}

    @property
    def attack_target(self) -> Point2:
        # Example logic from your code
        main_army_position: Point2 = self.mediator.get_position_of_main_squad(role=UnitRole.ATTACKING)

        if self.enemy_structures:
            closest_structure = cy_closest_to(main_army_position, self.enemy_structures).position
            if closest_structure.distance_to(main_army_position) > 25.0:
                return self.fallback_target
            return closest_structure.position
        elif self.time < 240.0:
            return self.enemy_start_locations[0]
        else:
            if self.is_visible(self.current_base_target):
                self.current_base_target = next(self.expansions_generator)
            return self.current_base_target

    @property
    def fallback_target(self) -> Point2:
        if self.is_visible(self.current_base_target):
            self.current_base_target = next(self.expansions_generator)
        return self.current_base_target

    async def on_start(self) -> None:
        await super().on_start()

        print("Game started")
        self.current_base_target = self.enemy_start_locations[0]

        # Sort expansions by distance to enemy
        self.expansion_locations_list.sort(
            key=lambda loc: loc.distance_to(self.enemy_start_locations[0])
        )
        self.scout_targets = self.expansion_locations_list

        self.natural_expansion: Point2 = await self.get_next_expansion()
        self._begin_attack_at_supply = 25.0

        self.expansions_generator = cycle([pos for pos in self.expansion_locations_list])
        self.freeflow: bool = self.minerals > 800 and self.vespene < 200

        print("Build Chosen:", self.build_order_runner.chosen_opening)

    async def on_step(self, iteration: int) -> None:
        await super().on_step(iteration)

        # Keep resource tags if needed
        self.resource_by_tag = {
            unit.tag: unit for unit in chain(self.mineral_field, self.gas_buildings)
        }

        # -- Basic references for our modules --
        main_army = self.mediator.get_units_from_role(role=UnitRole.ATTACKING)
        warp_prism = self.mediator.get_units_from_role(role=UnitRole.DROP_SHIP)
        scout_units = self.mediator.get_units_from_role(role=UnitRole.SCOUTING)

        # -------------
        # Reactions / Early Threat
        # -------------
        if not self._under_attack and not self.build_order_runner.build_completed:
            early_threat_sensor(self)

        if self._used_cheese_response:
            cheese_reaction(self)  # or you can handle in macro if you prefer

        if self._used_one_base_response:
            one_base_reaction(self)

        # -------------
        # Macro
        # -------------
        # Instead of a MacroManager, we call standalone functions
        await handle_macro(
            bot=self,
            iteration=iteration,
            main_army=main_army,
            warp_prism=warp_prism,
            scout_units=scout_units,
            attack_target=self.attack_target,
            freeflow=self.freeflow,
        )

        # Optional separate worker production logic
        worker_production(self)

        # -------------
        # Combat
        # -------------
        threat_detection(self, main_army)

        # If we want to directly control the main army or warp prism,
        # we can do so here (some folks do it in the macro or reaction modules).
        if self._commenced_attack and main_army:
            control_main_army(self, main_army, self.attack_target)

        if warp_prism:
            warp_prism_follower(self, warp_prism, main_army)

        # -------------
        # Scouting
        # -------------
        # If you want to do direct calls (some do it in macro or on its own):
        control_scout(self, scout_units, main_army)

        # -------------
        # Archon merging or fallback conditions
        # -------------
        if self.units(UnitTypeId.HIGHTEMPLAR).amount >= 2:
            for templar in self.units(UnitTypeId.HIGHTEMPLAR).ready:
                templar(AbilityId.MORPH_ARCHON)

        # Fallback if the build didn’t complete but we have too many minerals
        if self.minerals > 2500 and not self.build_order_runner.build_completed:
            self.build_order_runner.set_build_completed()

    async def on_unit_created(self, unit: Unit) -> None:
        await super().on_unit_created(unit)
        # Example assignment logic
        if unit.type_id in ALL_STRUCTURES or unit.type_id in WORKER_TYPES:
            return
        if unit.type_id == UnitTypeId.OBSERVER:
            self.mediator.assign_role(tag=unit.tag, role=UnitRole.SCOUTING)
        elif unit.type_id == UnitTypeId.WARPPRISM:
            self.mediator.assign_role(tag=unit.tag, role=UnitRole.DROP_SHIP)
            unit.move(self.natural_expansion.towards(self.game_info.map_center, 1))
        else:
            self.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)
            unit.attack(self.natural_expansion.towards(self.game_info.map_center, 2))

    async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
        await super().on_unit_took_damage(unit, amount_damage_taken)
        # Example building-cancel logic
        if unit.type_id in ALL_STRUCTURES:
            compare_health = max(50.0, unit.health_max * 0.09)
            if unit.health < compare_health:
                self.mediator.cancel_structure(structure=unit)

    async def on_end(self, game_result: Result) -> None:
        await super().on_end(game_result)

    # -------------
    # Utility Methods
    # -------------
    def get_total_supply(self, units: Units) -> float:
        # Example function if you used it in your macro code
        return sum(u._type_data.food_required for u in units)

    async def expand_to_next_location(self) -> None:
        if next_expand_loc := await self.get_next_expansion():
            if worker := self.mediator.select_worker(
                target_position=next_expand_loc,
                force_close=True,
            ):
                self.mediator.build_with_specific_worker(
                    worker=worker,
                    structure_type=UnitTypeId.NEXUS,
                    pos=next_expand_loc,
                )

    def use_overcharge(self, main_army: Units) -> bool:
        if self.total_health_shield_percentage >= 0.75:
            return False

        closest_nexus = None
        closest_distance = float("inf")
        for nexus in self.structures(UnitTypeId.NEXUS).ready:
            distance = main_army.center.distance_to(nexus.position)
            if distance < closest_distance:
                closest_distance = distance
                closest_nexus = nexus

        if not closest_nexus or closest_distance > 12:
            return False

        shield_batteries = self.structures(UnitTypeId.SHIELDBATTERY).closer_than(9, closest_nexus)
        if not shield_batteries:
            return False

        if closest_nexus.energy >= 50 and shield_batteries.ready:
            shield_battery = shield_batteries.closest_to(closest_nexus)
            closest_nexus(AbilityId.BATTERYOVERCHARGE_BATTERYOVERCHARGE, shield_battery)
            return True

        return False

    def defend_cannon_rush(self, enemy_probes, enemy_cannons):
        defend_worker_cannon_rush(self, enemy_probes, enemy_cannons)
