"""Custom behavior for Disruptor Nova ability.
This module implements the logic to fire the Disruptor Nova ability,
track its on-screen duration, and select optimal targets using enemy
and friendly positions. This serves as a basis for integration with the Ares combat framework.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional
from sc2.ids.ability_id import AbilityId

import traceback

from behaviors.combat.individual.combat_individual_behavior import CombatIndividualBehavior
from ares.behaviors.combat.individual import PathUnitToTarget
from ares.managers.manager_mediator import ManagerMediator
import numpy as np
from sc2.position import Point2
import math

from cython_extensions import cy_find_aoe_position


if TYPE_CHECKING:
    from ares import AresBot

class UseDisruptorNova(CombatIndividualBehavior):
    def __init__(self, mediator: ManagerMediator, bot: 'AresBot', position_update_frequency: int = 10):
        """Initialize with the given cooldown and nova duration in seconds."""
        self.cooldown = 21.4
        self.nova_duration = 2.1
        self.best_target_pos = None
        self.best_target_influence = 0
        self.unit = None
        self.original_target_pos = None
        self.target_pos = None
        self.executing = False
        self.frames_left = 0
        self.distance_left = 0.0
        self.mediator = mediator
        self.bot = bot  # Store bot instance
        self.should_debug_visuals = False  # Set to True for visual debugging
        
        # Do NOT store any references to grids here, get them fresh when needed
        self.influence_grid = None
        
        # Print initialization message
        print("DEBUG: UseDisruptorNova initialized")
        
        # Add movement cooldown tracking
        self.last_movement_frame = 0
        self.movement_cooldown = 5  # Only issue move commands every 5 frames
        
        # Add a timestamp to track when this Nova was started
        self.execution_start_time = 0
        self.max_execution_time = 10.0  # Maximum time in seconds a Nova should be executing
        
        # Position tracking and movement management
        self.max_distance_tracking_factor = 35.0  # Maximum distance to consider for tracking by the Nova
        self.frame_counter = 0  # To track frames since execution for position updates
        self.position_update_frequency = position_update_frequency  # How often to update position
        self.current_position = None  # Current Nova position
        self.initial_unit_position = None  # Position when the Nova was created
        self.last_target_update_time = 0  # Last time we updated the target position
    
    def can_use(self, disruptor_unit) -> bool:
        """Return True if the ability can be used based on its cooldown."""
        return (AbilityId.EFFECT_PURIFICATIONNOVA in disruptor_unit.abilities)

    def select_best_target(self, enemy_units: List['Unit'], friendly_units: List['Unit'], nova_manager) -> Optional[Point2]:
        """
        Select the best target position for the Nova based on enemy positions and influence grid.
        
        Args:
            enemy_units: List of enemy units to consider
            friendly_units: List of friendly units to avoid
            nova_manager: The NovaManager for exclusion mask and target registration
            
        Returns:
            Point2: The best target position, or None if no suitable target was found
        """
        try:
            # If no enemy units, there's no target
            if not enemy_units:
                print("DEBUG select_best_target: No enemy units to target")
                return None
            
            # Get the tactical ground grid - maintain property access without parentheses
            grid = self.mediator.get_tactical_ground_grid
            
            if grid is None:
                print("DEBUG select_best_target: No tactical grid available")
                return None
            
            # Nova explosion radius
            nova_radius = 1.5
            
            # Get exclusion mask from nova manager
            try:
                exclusion_mask = nova_manager.get_exclusion_mask(grid)
                print(f"DEBUG select_best_target: Got exclusion mask with {np.sum(exclusion_mask)} cells excluded")
            except Exception as e:
                print(f"DEBUG ERROR getting exclusion mask: {e}")
                exclusion_mask = None
            
            # Step 1: Use cy_find_aoe_position to get optimal AOE position
            optimal_position = cy_find_aoe_position(effect_radius=nova_radius, targets=enemy_units, bonus_tags=set())
            
            if optimal_position is None:
                print("DEBUG select_best_target: cy_find_aoe_position found no suitable position")
                return None
            
            # Step 2: Gather all candidate positions to evaluate
            candidate_positions = []
            
            # Add the optimal position from cy_find_aoe_position
            candidate_positions.append(Point2((optimal_position[0], optimal_position[1])))
            
            # Also consider individual enemy positions as candidates
            for enemy in enemy_units:
                pos = enemy.position
                if pos not in candidate_positions:
                    candidate_positions.append(pos)
            
            print(f"DEBUG select_best_target: Evaluating {len(candidate_positions)} candidate positions")
            
            # Step 3: Score each position based on influence and unit counts
            best_pos = None
            best_score = float('-inf')
            
            for pos in candidate_positions:
                try:
                    # Skip if position is out of grid bounds
                    grid_x, grid_y = int(pos.x), int(pos.y)
                    if not (0 <= grid_x < grid.shape[1] and 0 <= grid_y < grid.shape[0]):
                        continue
                    
                    # Skip if this position is in an exclusion zone
                    if exclusion_mask is not None:
                        if (0 <= grid_x < exclusion_mask.shape[1] and 
                            0 <= grid_y < exclusion_mask.shape[0] and 
                            exclusion_mask[grid_y, grid_x]):
                            continue
                    
                    # Get the influence value at this position
                    influence = grid[grid_y, grid_x]
                    
                    # Count and calculate influence of enemies within Nova radius
                    enemies_hit = 0
                    enemy_influence = 0
                    for enemy in enemy_units:
                        if pos.distance_to(enemy.position) <= nova_radius:
                            enemies_hit += 1
                            # Get influence at enemy position
                            enemy_x, enemy_y = int(enemy.position.x), int(enemy.position.y)
                            if (0 <= enemy_x < grid.shape[1] and 0 <= enemy_y < grid.shape[0]):
                                enemy_value = grid[enemy_y, enemy_x]
                                # Count influence only if it's higher than baseline (enemy units have higher influence)
                                if enemy_value > 200:
                                    enemy_influence += enemy_value - 200
                    
                    # Check and calculate negative impact from friendly units
                    friendly_hit = 0
                    friendly_influence = 0
                    for friendly in friendly_units:
                        if pos.distance_to(friendly.position) <= nova_radius:
                            friendly_hit += 1
                            # Get influence at friendly position
                            friendly_x, friendly_y = int(friendly.position.x), int(friendly.position.y)
                            if (0 <= friendly_x < grid.shape[1] and 0 <= friendly_y < grid.shape[0]):
                                friendly_value = grid[friendly_y, friendly_x]
                                # Count negative influence from hitting friendly units (friendly units have lower influence)
                                if friendly_value < 200:
                                    friendly_influence += 200 - friendly_value
                    
                    # Skip positions that would hit friendly units
                    if friendly_hit > 0:
                        continue
                    
                    # Calculate combined score:
                    # - Weight enemy count highly
                    # - Add enemy influence above baseline
                    # - Subtract friendly influence below baseline (though we already skip friendly fire)
                    score = (enemies_hit * 100) + enemy_influence - friendly_influence
                    
                    # Update best position if this is better
                    if score > best_score:
                        best_score = score
                        best_pos = pos
                        print(f"DEBUG select_best_target: Found better position at {best_pos} with score {best_score} "
                              f"(enemies: {enemies_hit}, enemy_influence: {enemy_influence}, friendly_influence: {friendly_influence})")
                except Exception as e:
                    print(f"DEBUG ERROR scoring position {pos}: {e}")
            
            # If we found a good position, return it
            if best_pos:
                print(f"DEBUG select_best_target: Selected target at {best_pos} with score {best_score}")
                return best_pos
            else:
                print("DEBUG select_best_target: No suitable target found")
                return None
        
        except Exception as e:
            print(f"DEBUG ERROR in select_best_target: {e}")
            return None

    def update_target_position(self, enemy_units: List['Unit'], friendly_units: List['Unit'], nova_manager):
        """
        Check if a better target has become available within Nova's remaining travel range.
        
        Args:
            enemy_units: List of enemy units to consider
            friendly_units: List of friendly units to avoid
            nova_manager: The NovaManager for exclusion mask and target registration
            
        Returns:
            bool: True if target was updated, False otherwise
        """
        if not self.best_target_pos or not self.unit:
            return False
            
        try:
            # Check if there's enough time to change course
            if self.frames_left < 5:  
                return False
            # Get the tactical grid
            grid = self.mediator.get_tactical_ground_grid
            # Get the current position of the Nova
            current_position = self.unit.position
            
            # Calculate the maximum distance the Nova can still travel
            max_travel_distance = nova_manager.nova_speed * (self.frames_left / 22.4)
            
            # Filter enemy units by distance to the Nova unit
            MAX_SEARCH_RADIUS = max_travel_distance + 2.0  # Add a small buffer
            nearby_enemies = [unit for unit in enemy_units if unit.position.distance_to(current_position) <= MAX_SEARCH_RADIUS]
            
            if not nearby_enemies:
                print("DEBUG update_target: No nearby enemy units within reach")
                return False
            
            # Get exclusion mask from nova_manager
            exclusion_mask = None
            if nova_manager and grid is not None:
                try:
                    exclusion_mask = nova_manager.get_exclusion_mask(grid)
                    print(f"DEBUG update_target: Got exclusion mask with {np.sum(exclusion_mask)} cells excluded")
                except Exception as e:
                    print(f"DEBUG ERROR getting exclusion mask in update_target: {e}")
            
            # Select the best target position
            new_target = self.select_best_target(nearby_enemies, friendly_units, nova_manager)
            
            if new_target and new_target != self.best_target_pos:
                print(f"DEBUG update_target: Target changed from {self.best_target_pos} to {new_target}")
                
                # Unregister old target if nova_manager is available
                if nova_manager:
                    try:
                        nova_manager.unregister_nova_target(self.best_target_pos)
                    except Exception as e:
                        print(f"DEBUG ERROR unregistering old target: {e}")
                
                # Update to new target
                self.best_target_pos = new_target
                
                # Register new target if nova_manager is available
                if nova_manager:
                    try:
                        nova_manager.register_nova_target(new_target)
                    except Exception as e:
                        print(f"DEBUG ERROR registering new target: {e}")
                
                return True
            
            return False
        except Exception as e:
            print(f"DEBUG ERROR in update_target_position: {e}")
            return False

    def load_info(self, unit):
        """Initialize nova tracking state when fired."""
        self.unit = unit
        self.frames_left = 48  # Reset to starting frame count
        self.distance_left = self.calculate_distance_left(unit.movement_speed)
        self.best_target_pos = unit.position  # Initial target position
        

    def execute(self, disruptor_unit, enemy_units: List['Unit'], friendly_units: List['Unit'], nova_manager=None):
        """Attempt to execute Disruptor Nova ability. Initializes nova state and simulates firing the nova.
        Returns self (the nova instance) if fired successfully, or None if not.
        
        Args:
            disruptor_unit: The Disruptor unit to fire the Nova
            enemy_units: List of enemy units to target
            friendly_units: List of friendly units to avoid
            nova_manager: Optional NovaManager for target coordination
        """
        # Check if ability can be used 
        if not self.can_use(disruptor_unit):
            print(f"DEBUG: Disruptor {disruptor_unit.tag} cannot use Nova - ability not ready")
            return None

        # Get exclusion mask from nova_manager if provided
        exclusion_mask = None
        if nova_manager:
            try:
                # Fresh call to get the grid
                grid = self.mediator.get_tactical_ground_grid
                if grid is None:
                    print("DEBUG: Tactical grid is None in execute")
                    return None
                
                # Then get the exclusion mask using the grid
                exclusion_mask = nova_manager.get_exclusion_mask(grid)
                print(f"DEBUG: Got exclusion mask with {np.sum(exclusion_mask)} cells excluded")
            except Exception as e:
                print(f"DEBUG ERROR getting exclusion mask: {e}")
                exclusion_mask = None

        # Select a target, considering exclusion zones
        print(f"DEBUG: Selecting target for Disruptor {disruptor_unit.tag}, {len(enemy_units)} enemy units, {len(friendly_units)} friendly units")
        target = None
        try:
            target = self.select_best_target(enemy_units, friendly_units, nova_manager)
        except Exception as e:
            print(f"DEBUG ERROR in select_best_target: {e}")
            
        if not target:
            print(f"DEBUG: No valid target found for Disruptor {disruptor_unit.tag}")
            return None
        else:
            print(f"DEBUG: Found target at {target}")
            
        # Register this target with the nova manager if provided
        target_registered = False
        if nova_manager:
            try:
                target_registered = nova_manager.register_nova_target(target)
                if not target_registered:
                    print(f"DEBUG: Failed to register target with NovaManager - continuing anyway")
                else:
                    print(f"DEBUG: Successfully registered target with NovaManager")
            except Exception as e:
                print(f"DEBUG ERROR registering target: {e}")
                # Continue anyway - we'll still try to fire the nova

        # Calculate which ability ID to use based on unit type (should be AbilityId.EFFECT_PURIFICATIONNOVA)
        ability_id = AbilityId.EFFECT_PURIFICATIONNOVA
        
        # Calculate maximum distance the Nova can travel during its lifetime
        nova_speed = 5.95  # Nova movement speed in game units per second
        nova_lifetime = 2.1  # Nova lifetime in seconds
        max_travel_distance = nova_speed * nova_lifetime
        
        # Calculate the current distance to the target
        current_distance = disruptor_unit.position.distance_to(target)
        
        # Debug info
        print(f"Target distance: {current_distance:.2f}, Max travel distance: {max_travel_distance:.2f}")
        
        # Check if the target is within range
        if current_distance <= max_travel_distance:
            # Target is within range, fire the Nova
            try:
                did_fire = disruptor_unit(ability_id, target)
            except Exception as e:
                print(f"DEBUG ERROR firing Nova: {e}")
                did_fire = False
        else:
            # Target is out of range, move the Disruptor closer
            # Find a position that moves toward the target but not all the way
            move_position = disruptor_unit.position.towards(target, 5.0)
            disruptor_unit.move(move_position)
            print(f"Target out of range. Moving Disruptor to: {move_position}")
            did_fire = False
            
        print(f"DEBUG: Disruptor execute result: {did_fire}")
        
        if did_fire:
            # On successful fire, initialize the nova instance and add to active novas
            self.best_target_pos = target
            self.frames_left = 48  # 2.1 seconds duration at 22.4 frames per sec
            return self
        else:
            # If firing failed, unregister the target only if we registered it successfully
            if nova_manager and target_registered:
                try:
                    nova_manager.unregister_nova_target(target)
                except Exception as e:
                    print(f"DEBUG ERROR unregistering unused target: {e}")
            return None

    
    def select_best_target_pos(self, disruptor_unit, enemy_units, friendly_units=None, exclusion_mask=None) -> Point2:
        """
        Select the best position to target with the Nova.
        
        Args:
            disruptor_unit: The disruptor unit to use the nova with
            enemy_units: List of enemy units to consider as targets
            friendly_units: (Optional) List of friendly units to avoid damaging
            nova_manager: (Optional) The NovaManager instance
            
        Returns:
            Point2: The best target position, or None if no valid target
        """
        try:
            # Convert to empty lists if None
            if enemy_units is None:
                enemy_units = []
            if friendly_units is None:
                friendly_units = []
                
            # Filter out non-attackable units
            valid_enemy_units = [
                unit for unit in enemy_units
                if unit.is_visible and not unit.is_flying and not unit.is_structure
            ]
            
            if not valid_enemy_units:
                print(f"DEBUG: No valid enemy units to target with Nova")
                return None
                
            # Filter enemies that are in range
            nova_range = 15.0  # Maximum reasonable range for Nova ability
            in_range_enemies = [
                unit for unit in valid_enemy_units
                if unit.distance_to(disruptor_unit) < nova_range
            ]
            
            if not in_range_enemies:
                print(f"DEBUG: No enemy units in range of disruptor")
                return None
                
            # Find the best cluster of enemies
            best_enemy = None
            best_score = -1
            
            for enemy in in_range_enemies:
                # Calculate base score from distance to disruptor (closer is better)
                # Use inverse distance so closer enemies have higher scores
                distance_to_disruptor = enemy.distance_to(disruptor_unit)
                if distance_to_disruptor < 1.0:
                    distance_to_disruptor = 1.0  # Avoid division by zero
                
                base_score = 100.0 / distance_to_disruptor
                
                # Count nearby enemies to this enemy for cluster score
                nearby_count = sum(1 for other in in_range_enemies if other.distance_to(enemy) < 2.5)
                cluster_score = nearby_count * 50.0  # Each nearby enemy adds 50 points
                
                # Combine scores
                total_score = base_score + cluster_score
                
                if total_score > best_score:
                    best_score = total_score
                    best_enemy = enemy
            
            # If we found a good target, return its position
            if best_enemy:
                return best_enemy.position
                
            return None
            
        except Exception as e:
            print(f"DEBUG ERROR selecting best target for Nova: {e}")
            return None

    def update_info(self):
        """Update nova tracking state each step."""
        self.frames_left -= 1  # Decrement frame count
        self.distance_left = self.calculate_distance_left(self.unit.movement_speed)
    
    def calculate_distance_left(self, unit_speed: float) -> float:
        """Calculate remaining distance based on unit movement speed and frames left.

        Uses the constant 22.4 frames per second to convert speed into distance per frame.
        """
        return self.frames_left * (unit_speed / 22.4)

    
    def run_step(self, enemy_units: List['Unit'], friendly_units: List['Unit'], nova_manager=None):
        """Execute one step for this nova. Reduces the frame counter and possibly moves the nova.
        Returns True if the nova is still active, False if it has expired.
        
        Args:
            enemy_units: List of enemy units to avoid hitting
            friendly_units: List of friendly units to avoid hitting
            nova_manager: Optional NovaManager for target updating
        """
    
        if self.frames_left <= 0:
            return False
        print(f"DEBUG: Nova at {self.unit.position} has {self.frames_left} frames left")

        # Update the frame counter
        self.frames_left -= 1
        
        # Check if we should update our target - do this every frame to be more responsive
        if nova_manager:
            self.update_target_position(enemy_units, friendly_units, nova_manager)

        # If a valid target was found and it differs from the current position, command the nova to move
        if self.best_target_pos is not None and self.best_target_pos != self.unit.position:
            #TODO need to find a way for nova to pathfind to the best target position
            self.unit.move(self.best_target_pos)
            print(f"Moving Nova to target: {self.best_target_pos}")
        
        # Optionally, output debug information
        # Continue running until the nova expires
        return self.frames_left > 0