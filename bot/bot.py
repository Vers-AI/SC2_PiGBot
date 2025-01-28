# bot.py

from typing import Optional
from itertools import cycle, chain
import numpy as np

# Ares imports (framework-specific)
from ares import AresBot
from ares.consts import ALL_STRUCTURES, WORKER_TYPES, UnitRole

from ares.managers.manager_mediator import ManagerMediator
from ares.managers.manager import Manager
from ares.managers.squad_manager import UnitSquad


# Cython or custom references
from cython_extensions import (
    cy_closest_to,
)

# SC2-related imports
from sc2.data import Result
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units



# Modular imports for separated concerns
from bot.hub.macro import handle_macro, worker_production
from bot.hub.combat import (
    attack_target,
    control_main_army,
    threat_detection,
    warp_prism_follower,
    handle_attack_toggles
)
from bot.hub.scouting import control_scout
from ares.behaviors.macro import Mining
from bot.hub.reactions import (
    early_threat_sensor,
    cheese_reaction,
    defend_worker_cannon_rush,
    one_base_reaction
)

# These are units to ignore when picking targets or performing certain queries
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


class PiG_Bot(AresBot):
    """
    Main class for our Protoss bot, leveraging the Ares framework.
    Logic is organized into separate modules for macro, combat, scouting, and reactions.
    """

    current_base_target: Point2
    expansions_generator: cycle
    _begin_attack_at_supply: float

    def __init__(self, game_step_override: Optional[int] = None):
        """
        Initializes the bot with various flags and references needed by Ares.
        """
        super().__init__(game_step_override)

        # State tracking
        self.unit_roles = {}
        self.scout_targets = {}
        self.bases = {}
        self.total_health_shield_percentage = 1.0

        # Flags for in-game logic
        self._commenced_attack = False
        self._used_cheese_response = False
        self._used_one_base_response = False
        self._under_attack = False
        self._cheese_reaction_completed = False
        self._one_base_reaction_completed = False
        self._is_building = False



    async def on_start(self) -> None:
        """
        Runs at the start of the game. Sets up expansions, initial flags,
        and calculates where to take expansions first.
        """
        await super().on_start()
        print("Game started")

        self.current_base_target = self.enemy_start_locations[0]

        # Sort expansions by proximity to the enemy's start
        self.expansion_locations_list.sort(
            key=lambda loc: loc.distance_to(self.enemy_start_locations[0])
        )
        self.scout_targets = self.expansion_locations_list

        # Reserve expansions and set flags
        self.natural_expansion = await self.get_next_expansion()
        self._begin_attack_at_supply = 25.0
        self.expansions_generator = cycle(self.expansion_locations_list)
        self.freeflow = self.minerals > 800 and self.vespene < 200

        print("Build Chosen:", self.build_order_runner.chosen_opening)

    async def on_step(self, iteration: int) -> None:
        """
        Main loop executed each game step. Calls out to macro, combat, scouting,
        and reaction modules to handle behavior.
        """
        await super(PiG_Bot, self).on_step(iteration)
        self.register_behavior(Mining()) #ares Mining 

        # Retrieve roles
        main_army = self.mediator.get_units_from_role(role=UnitRole.ATTACKING)
        warp_prism = self.mediator.get_units_from_role(role=UnitRole.DROP_SHIP)
        scout_units = self.mediator.get_units_from_role(role=UnitRole.SCOUTING)

        # Create Squad
        squads: list[UnitSquad] = self.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=15)

        # If not under attack and build order isn't done, do an early threat check
        if not self._under_attack and not self.build_order_runner.build_completed:
            early_threat_sensor(self)

        # If cheese or one-base flags are set, handle them
        if self._used_cheese_response:
            cheese_reaction(self)
        if self._used_one_base_response:
            one_base_reaction(self)

        # Macro calls
        await handle_macro(
            bot=self,
            iteration=iteration,
            main_army=main_army,
            warp_prism=warp_prism,
            scout_units=scout_units,
            freeflow=self.freeflow,
        )
        worker_production(self)

        # Run combat-oriented threat detection
        threat_detection(self, main_army)

        # Handle attack toggles if main_army exists
        if main_army:
            self.main_army_position = self.mediator.get_position_of_main_squad(role=UnitRole.ATTACKING)
            handle_attack_toggles(self, main_army, attack_target(self, main_army_position=self.main_army_position))

        # Optionally control main army or warp prism outside macro
        if self._commenced_attack and main_army:
            control_main_army(self, main_army, attack_target(self, main_army_position=self.main_army_position), squads)
        if warp_prism:
            warp_prism_follower(self, warp_prism, main_army)

        # Scouting actions
        control_scout(self, scout_units, main_army)

        # Merge Archons if we have 2 or more High Templar
        if self.units(UnitTypeId.HIGHTEMPLAR).amount >= 2:
            for templar in self.units(UnitTypeId.HIGHTEMPLAR).ready:
                templar(AbilityId.MORPH_ARCHON)

        # Fail-safe if build order still not done but we have too many minerals
        if self.minerals > 2500 and not self.build_order_runner.build_completed:
            self.build_order_runner.set_build_completed()

    async def on_unit_created(self, unit: Unit) -> None:
        """
        Called whenever a new unit spawns. Assign roles based on type.
        """
        await super().on_unit_created(unit)

        if unit.type_id in ALL_STRUCTURES or unit.type_id in WORKER_TYPES:
            return

        if unit.type_id == UnitTypeId.OBSERVER:
            self.mediator.assign_role(tag=unit.tag, role=UnitRole.SCOUTING)
            return
        if unit.type_id == UnitTypeId.WARPPRISM:
            self.mediator.assign_role(tag=unit.tag, role=UnitRole.DROP_SHIP)
            unit.move(self.natural_expansion.towards(self.game_info.map_center, 1))
            return

        # Default: Attacking role
        self.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)
        unit.attack(self.natural_expansion.towards(self.game_info.map_center, 2))

    async def on_unit_took_damage(self, unit: Unit, amount_damage_taken: float) -> None:
        """
        Allows us to cancel building structures if they're badly damaged, 
        preventing resource waste.
        """
        await super().on_unit_took_damage(unit, amount_damage_taken)

        if unit.type_id not in ALL_STRUCTURES:
            return
        compare_health = max(50.0, unit.health_max * 0.09)
        if unit.health < compare_health:
            self.mediator.cancel_structure(structure=unit)

    async def on_end(self, game_result: Result) -> None:
        """
        Called at the end of the game. This can be used for post-match analysis.
        """
        await super().on_end(game_result)

    # -------------------------------------
    # Utility / Additional Methods
    # -------------------------------------
    async def expand_to_next_location(self) -> None:
        """
        Triggered when we decide to expand to another base location.
        Chooses a worker and instructs it to build a Nexus.
        """
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
        """
        Checks conditions for Shield Battery Overcharge if army is near 
        a Nexus with enough energy.
        """
        if self.total_health_shield_percentage >= 0.75:
            return False

        # Find nearest Nexus
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
            battery = shield_batteries.closest_to(closest_nexus)
            closest_nexus(AbilityId.BATTERYOVERCHARGE_BATTERYOVERCHARGE, battery)
            return True
        return False

    def defend_cannon_rush(self, enemy_probes, enemy_cannons):
        """
        Delegates to a function in reactions.py to handle a cannon rush scenario 
        by pulling and microing worker units.
        """
        defend_worker_cannon_rush(self, enemy_probes, enemy_cannons)

    @property
    def Standard_Army(self) -> dict:
        """
        Composition for general, non-cheese play.
        """
        return {
            UnitTypeId.IMMORTAL: {"proportion": 0.2, "priority": 2},
            UnitTypeId.COLOSSUS: {"proportion": 0.1, "priority": 3},
            UnitTypeId.HIGHTEMPLAR: {"proportion": 0.45, "priority": 1},
            UnitTypeId.ZEALOT: {"proportion": 0.25, "priority": 0},
        }

    @property
    def cheese_defense_army(self) -> dict:
        """
        Composition specifically for defending early aggression or cheese.
        """
        return {
            UnitTypeId.ZEALOT: {"proportion": 0.5, "priority": 0},
            UnitTypeId.STALKER: {"proportion": 0.4, "priority": 1},
            UnitTypeId.ADEPT: {"proportion": 0.1, "priority": 2},
        }
