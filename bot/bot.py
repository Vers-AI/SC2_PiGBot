# bot.py

from typing import Optional
from itertools import cycle, chain
import numpy as np

# Ares imports (framework-specific)
from ares import AresBot
from ares.consts import ALL_STRUCTURES, WORKER_TYPES, UnitRole
# from ares.managers.unit_manager import UnitManager
from ares.managers.squad_manager import UnitSquad
from ares.managers.manager_mediator import ManagerMediator
from map_analyzer import MapData
from ares.behaviors.macro.restore_power import RestorePower


# SC2-related imports
from sc2.data import Result
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from sc2.unit import Unit
from sc2.units import Units
from sc2.data import Race



# Modular imports for separated concerns
from bot.managers.macro import handle_macro
from bot.managers.reactions import defend_cannon_rush, defend_worker_rush, early_threat_sensor, cheese_reaction, one_base_reaction
from bot.managers.combat import (
    attack_target,
    control_main_army,
    threat_detection,
    warp_prism_follower,
    handle_attack_toggles,
    regroup_army,
    gatekeeper_control
)
from bot.managers.scouting import control_scout
from ares.behaviors.macro import Mining
from bot.managers.reactions import (
    early_threat_sensor,
    cheese_reaction,
    defend_cannon_rush,
    one_base_reaction
)
#debugs
from bot.utilities.use_disruptor_nova import UseDisruptorNova
from map_analyzer import MapData
from bot.utilities.nova_manager import NovaManager




class PiG_Bot(AresBot):
    """
    Main class for our Protoss bot, leveraging the Ares framework.
    Logic is organized into separate modules for macro, combat, scouting, and reactions.
    """

    current_base_target: Point2
    expansions_generator: cycle

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
        self.enemy_army = None
        self.own_army = None
        # Game state: 0 = early game, 1 = mid game, 2 = late game
        self.game_state = 0
        self.early_game_threshold = 360  # 6 minutes in seconds
        self.mid_game_threshold = 720    # 12 minutes in seconds
        
        # Observer assignments
        self.observer_assignments = {
            "primary": None,      # The first observer (enemy nat)
            "army": None,         # The second observer (follows army)
            "patrol": [],         # Additional observers (map locations)
        }
        self.observer_targets = {}  # Maps observer.tag -> current target

        # Flags for in-game logic
        self._commenced_attack = False
        self._used_cheese_response = False
        self._used_one_base_response = False
        self._under_attack = False
        self._cheese_reaction_completed = False
        self._one_base_reaction_completed = False
        self._not_worker_rush = True
        self._cannon_rush_response = False
        self._is_building = False
        
        # Cannon rush specific flags
        self._cannon_rush_active = False
        self._cannon_rush_completed = False
        self._cannon_rush_cleanup_timer = None

        # Debug flags
        self.debug = False



    async def on_start(self) -> None:
        """
        Runs at the start of the game. Sets up expansions, initial flags,
        and calculates where to take expansions first.
        """
        await super().on_start()
        print("Game started")

        # Debug on start
        self.map_data: MapData  = self.mediator.get_map_data_object
        
        # Debug control - set to True to enable debug output for disruptor nova system
        debug_disruptor_nova = False
        
        self.use_disruptor_nova = UseDisruptorNova(mediator=self.mediator, bot=self, debug_output=debug_disruptor_nova)
        self.nova_manager = NovaManager(bot=self, mediator=self.mediator, debug_output=debug_disruptor_nova)  # Initialize the NovaManager

        self.current_base_target = self.enemy_start_locations[0]

        # Sort expansions by proximity to the enemy's start
        self.expansion_locations_list.sort(
            key=lambda loc: loc.distance_to(self.enemy_start_locations[0])
        )
        self.scout_targets = self.expansion_locations_list
        


        # Reserve expansions and set flags
        self.natural_expansion: Point2 = self.mediator.get_own_nat
        

        self.expansions_generator = cycle(self.expansion_locations_list)
        self.freeflow = self.minerals > 800 and self.vespene < 200

        if self.enemy_race == Race.Zerg:
            self.gatekeeping_pos = self.mediator.get_pvz_nat_gatekeeping_pos
            if self.gatekeeping_pos is not None:
                self.rally_point = self.gatekeeping_pos.towards(self.natural_expansion, 5)
            else:
                self.rally_point = self.natural_expansion.towards(self.game_info.map_center, 5)

        else:
            self.rally_point = self.natural_expansion.towards(self.game_info.map_center, 5)


        print("Build Chosen:", self.build_order_runner.chosen_opening)
        print("the enemy race is:", self.enemy_race)

    async def on_step(self, iteration: int) -> None:
        """
        Main loop executed each game step. Calls out to macro, combat, scouting,
        and reaction modules to handle behavior.
        """
        await super(PiG_Bot, self).on_step(iteration)
        self.register_behavior(Mining(keep_safe=self._not_worker_rush)) #ares Mining 

        self.enemy_army = self.mediator.get_cached_enemy_army

        # Retrieve roles
        main_army = self.mediator.get_units_from_role(role=UnitRole.ATTACKING)
        warp_prism = self.mediator.get_units_from_role(role=UnitRole.DROP_SHIP)
        scout_units = self.mediator.get_units_from_role(role=UnitRole.SCOUTING)
        gatekeeper = self.mediator.get_units_from_role(role=UnitRole.GATE_KEEPER)


        # Create Squad
        squads: list[UnitSquad] = self.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=15)

        # Always run combat-oriented threat detection first
        # This ensures we're always responding to immediate threats regardless of build order status
        if main_army:  # Only run detection if we have an army to use for defense
            threat_detection(self, main_army)
            self.own_army = self.mediator.get_own_army


        # Early game logic
        if not self.build_order_runner.build_completed:
            self.register_behavior(RestorePower()) # Restore power to depowered buildings
            if not self._under_attack:  # Still use early_threat_sensor for cheese detection
                early_threat_sensor(self)    
            # If cheese or one-base flags are set, handle them
            if self._used_cheese_response:
                if not self._not_worker_rush:
                    # Handle worker rush defense
                    defend_worker_rush(self)
                elif self._cannon_rush_response:
                    # Handle cannon rush defense
                    defend_cannon_rush(self)
                else:
                    # Handle other cheese responses
                    cheese_reaction(self)
            elif self._used_one_base_response:
                one_base_reaction(self)
        else:
            # Macro calls (only run if build order is complete)
            await handle_macro(
                bot=self,
                iteration=iteration,
                main_army=main_army,
                warp_prism=warp_prism,
                scout_units=scout_units,
                freeflow=self.freeflow,
            )

        # Handle attack toggles if main_army exists
        if main_army:
            self.main_army_position = self.mediator.get_position_of_main_squad(role=UnitRole.ATTACKING)
            handle_attack_toggles(self, main_army, attack_target(self, main_army_position=self.main_army_position))

        # Optionally control main army or warp prism outside macro
        if self._commenced_attack and main_army:
            control_main_army(self, main_army, attack_target(self, main_army_position=self.main_army_position), squads)

        # Warp Prism following main army
        warp_prism_follower(self, warp_prism, main_army)

        # Check if the main army needs to regroup when not under attack or engaged in attack
        if not self._under_attack and not self._commenced_attack and main_army:
            regroup_army(self, main_army)

        # Scouting actions
        from bot.managers.scouting import control_observers
        observers = self.units(UnitTypeId.OBSERVER)
        control_observers(self, observers, main_army)

        # Merge Archons if we have 2 or more High Templar
        if self.units(UnitTypeId.HIGHTEMPLAR).amount >= 2:
            for templar in self.units(UnitTypeId.HIGHTEMPLAR).ready:
                templar(AbilityId.MORPH_ARCHON)

        # Fail-safe if build order still not done but we have too many minerals
        if self.minerals > 2800 and not self.build_order_runner.build_completed:
            self.build_order_runner.set_build_completed()
        
        # Update game state based on game time
        current_time = self.time
        if current_time >= self.mid_game_threshold:
            self.game_state = 2  # late game
        elif current_time >= self.early_game_threshold:
            self.game_state = 1  # mid game
            if gatekeeper:
                for zealot in gatekeeper:
                    self.mediator.clear_role(tag=zealot.tag)
                    self.mediator.assign_role(tag=zealot.tag, role=UnitRole.ATTACKING)
        else:
            self.game_state = 0  # early game
            if self.enemy_race == Race.Zerg:
                if not gatekeeper:
                    zealots = self.units(UnitTypeId.ZEALOT).ready
                    if zealots:
                        tag = zealots.first.tag
                        self.mediator.clear_role(tag=tag)
                        self.mediator.assign_role(tag=tag, role=UnitRole.GATE_KEEPER)

                elif not self._commenced_attack:
                    gatekeeper_control(self, gatekeeper)
                else:
                    for zealot in gatekeeper:
                        self.mediator.clear_role(tag=zealot.tag)
                        self.mediator.assign_role(tag=zealot.tag, role=UnitRole.ATTACKING)
                    
                    
    
    async def on_building_construction_complete(self, unit: Unit) -> None:
        """
        Called whenever a new building is completed. 
        """
        await super().on_building_construction_complete(unit)
        if unit.type_id == UnitTypeId.GATEWAY:
            if self.rally_point:
                unit(AbilityId.RALLY_BUILDING, self.rally_point)
            

    async def on_unit_created(self, unit: Unit) -> None:
        """
        Called whenever a new unit spawns. Assign roles based on type.
        """
        await super().on_unit_created(unit)
        if unit.type_id in ALL_STRUCTURES or unit.type_id in WORKER_TYPES:
            return

        if unit.type_id == UnitTypeId.OBSERVER:
            # First Observer - primary scout (enemy natural in mid-game)
            if self.observer_assignments["primary"] is None:
                self.observer_assignments["primary"] = unit.tag
                self.mediator.assign_role(tag=unit.tag, role=UnitRole.SCOUTING)
            # Second Observer - army follower
            elif self.observer_assignments["army"] is None:
                self.observer_assignments["army"] = unit.tag
                self.mediator.assign_role(tag=unit.tag, role=UnitRole.CONTROL_GROUP_EIGHT)
            # Additional Observers - patrol key map positions
            else:
                self.observer_assignments["patrol"].append(unit.tag)
                self.mediator.assign_role(tag=unit.tag, role=UnitRole.CONTROL_GROUP_NINE)
            return
            
        if unit.type_id == UnitTypeId.WARPPRISM:
            self.mediator.assign_role(tag=unit.tag, role=UnitRole.DROP_SHIP)
            unit.move(Point2(self.rally_point))
            return
        

        if unit.type_id == UnitTypeId.DISRUPTORPHASED:
            # When a DISRUPTORPHASED unit (the nova) is created, send it to the NovaManager
            self.nova_manager.add_nova(unit)
            return

        # Default: Attacking role
        self.mediator.assign_role(tag=unit.tag, role=UnitRole.ATTACKING)
        unit.attack(Point2(self.rally_point))

    async def on_unit_destroyed(self, unit_tag: int) -> None:
        """Track when units get destroyed."""
        await super(PiG_Bot, self).on_unit_destroyed(unit_tag)
        
        # Handle observer reassignment if destroyed
        if unit_tag == self.observer_assignments.get("primary"):
            self.observer_assignments["primary"] = None
            # Try to promote a patrol observer if available
            if self.observer_assignments["patrol"]:
                self.observer_assignments["primary"] = self.observer_assignments["patrol"].pop(0)
                # Update the role if the unit still exists
                if self.observer_assignments["primary"] in [unit.tag for unit in self.units]:
                    self.mediator.clear_role(tag=self.observer_assignments["primary"])
                    self.mediator.assign_role(tag=self.observer_assignments["primary"], role=UnitRole.SCOUTING)
        
        elif unit_tag == self.observer_assignments.get("army"):
            self.observer_assignments["army"] = None
            # Try to promote a patrol observer if available
            if self.observer_assignments["patrol"]:
                self.observer_assignments["army"] = self.observer_assignments["patrol"].pop(0)
                # Update the role if the unit still exists
                if self.observer_assignments["army"] in [unit.tag for unit in self.units]:
                    self.mediator.clear_role(tag=self.observer_assignments["army"])
                    self.mediator.assign_role(tag=self.observer_assignments["army"], role=UnitRole.CONTROL_GROUP_EIGHT)
        
        elif unit_tag in self.observer_assignments.get("patrol", []):
            self.observer_assignments["patrol"].remove(unit_tag)
            
        # Clean up observer targets if needed
        if unit_tag in self.observer_targets:
            del self.observer_targets[unit_tag]

    async def on_unit_type_changed(self, unit: Unit, previous_type: UnitTypeId) -> None:
        """Called when a unit changes type, like Disruptor firing a Nova."""
        await super(PiG_Bot, self).on_unit_type_changed(unit, previous_type)
        
        # Detect when a Disruptor fires a Nova (it changes type temporarily)
        if previous_type == UnitTypeId.DISRUPTOR and unit.type_id == UnitTypeId.DISRUPTORPHASED:
            # Add this nova to our manager for tracking
            self.nova_manager.add_nova(unit)
            
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
