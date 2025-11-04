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
from bot.managers.macro import handle_macro, get_optimal_gas_workers, get_freeflow_mode
from bot.managers.structure_manager import use_chronoboost
from bot.managers.reactions import defend_cannon_rush, defend_worker_rush, early_threat_sensor, cheese_reaction, one_base_reaction, threat_detection
from bot.combat import (
    control_main_army,
    warp_prism_follower,
    handle_attack_toggles,
    attack_target,
    gatekeeper_control,
    manage_defensive_unit_roles
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
from bot.utilities.performance_monitor import PerformanceMonitor
from bot.utilities.game_report import print_end_game_report
# # Wall manager removed - use generate_wall_placements.py to create wall data
# from bot.utilities.natural_wall_manager import NaturalWallManager  # Temporarily disabled




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
        self.own_army: Units = Units([], self)
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
        self._rush_time_seconds = 0.0  # Rush time calculated in on_start
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
        self.debug = False  # Enable debug output for targeting analysis
        
        # Target persistence for stable attack behavior
        self.current_attack_target = None
        self.target_lock_distance = 25.0  # Don't switch targets unless new one is 25+ units closer
        
        # Combat engagement tracking
        self._attack_commenced_time = 0.0  # Time when attack was initiated
        self._squad_engagement_tracker = {}  # Per-squad engagement decisions for tactical control
        
        # Performance monitoring (tracks SQ and other efficiency metrics)
        self.performance_monitor = PerformanceMonitor(sample_interval=22, alpha=0.1)
        
        # Natural wall management system
        self.wall_manager = None  # Will be initialized in on_start
        
        # Gas worker management
        self._gas_worker_toggle = True  # Toggle for gas mining on/off



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

        # Wall management moved to separate generate_wall_placements.py tool
        # Run: python generate_wall_placements.py to create wall data
        # self.wall_manager = NaturalWallManager(self)
        # opponent_race = await self.wall_manager.get_opponent_race_string()
        # await self.wall_manager.ensure_map_wall_exists(opponent_race)

        self.current_base_target = self.enemy_start_locations[0]

        # Sort expansions by proximity to the enemy's start
        self.expansion_locations_list.sort(
            key=lambda loc: loc.distance_to(self.enemy_start_locations[0])
        )
        self.scout_targets = self.expansion_locations_list
        


        # Reserve expansions and set flags
        self.natural_expansion: Point2 = self.mediator.get_own_nat
        print("Natural Expansion:", self.natural_expansion)
        print("Enemy Start:", self.mediator.get_enemy_nat)

        self.expansions_generator = cycle(self.expansion_locations_list)

        if self.enemy_race in {Race.Zerg, Race.Random}:
            if self.mediator.get_pvz_nat_gatekeeping_pos is not None:
                self.gatekeeping_pos = self.mediator.get_pvz_nat_gatekeeping_pos
            else:
                self.gatekeeping_pos = self.natural_expansion.towards(self.game_info.map_center, 6)
            
            if self.gatekeeping_pos is not None:
                self.rally_point = self.gatekeeping_pos.towards(self.natural_expansion, 5)
            else:
                self.rally_point = self.natural_expansion.towards(self.game_info.map_center, 5)

        else:
            self.rally_point = self.natural_expansion.towards(self.game_info.map_center, 5)
        
        print("Build Chosen:", self.build_order_runner.chosen_opening)
        print("the enemy race is:", self.enemy_race)
        
        # Compute rush distance tier for ling rush detection (used in reactions.py)
        from bot.utilities.rush_detection import compute_rush_distance_tier
        self.rush_distance_tier = compute_rush_distance_tier(self)
        rush_time_str = f"{self._rush_time_seconds:.1f}s" if self._rush_time_seconds > 0 else "unknown (pathfind failed)"
        print(f"Rush distance tier: {self.rush_distance_tier} ({rush_time_str} rush time)")

    async def on_step(self, iteration: int) -> None:
        """
        Main loop executed each game step. Calls out to macro, combat, scouting,
        and reaction modules to handle behavior.
        """
        await super(PiG_Bot, self).on_step(iteration)
        
        # Update performance metrics (SQ tracking)
        self.performance_monitor.update(iteration, self)
        
        # Dynamic gas worker management (logic in macro.py)
        self.register_behavior(Mining(
            workers_per_gas=get_optimal_gas_workers(self),
            keep_safe=self._not_worker_rush
        )) 

        self.enemy_army = self.mediator.get_cached_enemy_army

        # Retrieve roles (initial fetch)
        main_army = self.mediator.get_units_from_role(role=UnitRole.ATTACKING)
        warp_prism = self.mediator.get_units_from_role(role=UnitRole.DROP_SHIP)
        scout_units = self.mediator.get_units_from_role(role=UnitRole.SCOUTING)
        gatekeeper = self.mediator.get_units_from_role(role=UnitRole.GATE_KEEPER)

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
                freeflow=get_freeflow_mode(self),  # Dynamic calculation
            )
            
            # Use chronoboost on production/research structures
            use_chronoboost(self)

        # Manage defensive unit roles (return them to attacking when threats are cleared)
        manage_defensive_unit_roles(self)
        
        # CRITICAL: Refresh main_army after role management to get current state
        main_army = self.mediator.get_units_from_role(role=UnitRole.ATTACKING)
        
        # Create squads ONCE after all role management is complete (ARES pattern)
        from bot.constants import ATTACKING_SQUAD_RADIUS
        squads: list[UnitSquad] = self.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=ATTACKING_SQUAD_RADIUS)
        
        # Determine attack target and handle attack decision logic
        if main_army and squads:
            self.main_army_position = self.mediator.get_position_of_main_squad(role=UnitRole.ATTACKING)
            # handle_attack_toggles decides target (attack/retreat/rally) and sets _commenced_attack flag
            final_target = handle_attack_toggles(self, main_army, attack_target(self, main_army_position=self.main_army_position))
            # Always control when we have squads - target determines behavior (attack/retreat/rally)
            # This differs from OLD code which only controlled when _commenced_attack==True,
            # but OLD code called control_main_army inside handle_attack_toggles for retreats.
            # Net effect: army is always controlled when squads exist (same total behavior)
            control_main_army(self, main_army, final_target, squads)
        elif main_army:
            # Fallback: use main army center if no squads can be formed
            self.main_army_position = main_army.center
            # Still run decision logic to set flags
            handle_attack_toggles(self, main_army, attack_target(self, main_army_position=self.main_army_position))

        # Warp Prism following main army
        warp_prism_follower(self, warp_prism, main_army)

        # Army cohesion is now handled proactively by the main squad coordination system

        # Scouting actions
        from bot.managers.scouting import control_observers
        observers = self.units(UnitTypeId.OBSERVER)
        control_observers(self, observers, main_army)

        # Merge Archons if we have 2 or more High Templar
        if self.units(UnitTypeId.HIGHTEMPLAR).amount >= 2:
            for templar in self.units(UnitTypeId.HIGHTEMPLAR).ready:
                templar(AbilityId.MORPH_ARCHON)

       
        
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
             # Fail-safe: Force complete build if banking too many minerals
            if self.minerals > 800 and not self.build_order_runner.build_completed:
                self.build_order_runner.set_build_completed()
                print(f"Build order force-completed at {self.time:.1f}s due to high minerals")
            self.game_state = 0  # early game
            if self.enemy_race in {Race.Zerg, Race.Random}:
                if not gatekeeper:
                    zealots = self.units(UnitTypeId.ZEALOT).ready
                    if zealots:
                        tag = zealots.first.tag
                        self.mediator.clear_role(tag=tag)
                        self.mediator.assign_role(tag=tag, role=UnitRole.GATE_KEEPER)

                else:
                    # Keep gatekeeper active throughout early game regardless of attack status
                    gatekeeper_control(self, gatekeeper)
                    
                    
    
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
        Called at the end of the game - prints performance report and logs rush detection data.
        """
        print_end_game_report(
            performance_monitor=self.performance_monitor,
            game_result=game_result,
            game_time=self.time,
            idle_worker_time=self.state.score.idle_worker_time,
            idle_production_time=self.state.score.idle_production_time
        )
        
        # Log rush detection results for ML training
        from bot.utilities.rush_detection import log_rush_detection_result
        log_rush_detection_result(self, game_result)

   
    

   
