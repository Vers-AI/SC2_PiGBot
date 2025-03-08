"""Custom behavior for Disruptor Nova ability.
This module implements the logic to fire the Disruptor Nova ability,
track its on-screen duration, and select optimal targets using enemy
and friendly positions. This serves as a basis for integration with the Ares combat framework.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional
from sc2.data import UnitTypeId
from sc2.ids.ability_id import AbilityId

import traceback

from behaviors.combat.individual.combat_individual_behavior import CombatIndividualBehavior
from ares.behaviors.combat.individual import PathUnitToTarget
from ares.managers.manager_mediator import ManagerMediator
import numpy as np
from sc2.position import Point2
import math

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
        Select the best position to target with a Disruptor Nova ability.
        Uses influence grid to find the highest value point within range of the disruptor.
        
        Args:
            enemy_units: List of enemy units to consider targeting
            friendly_units: List of friendly units to avoid damaging
            nova_manager: NovaManager instance for target coordination
            
        Returns:
            Point2: The recommended target position, or None if no good position found
        """
        try:
            nova_radius = 1.5
            disruptor_max_range = 13.0  # Maximum range a disruptor can fire Nova
            
            # First, ensure we have some units to target
            if not enemy_units:
                print("DEBUG: No enemy units available to target")
                return None
                
            print(f"DEBUG: Found {len(enemy_units)} enemy units to consider")
                
            # Get the disruptor unit to determine position
            disruptor_unit = None
            for unit in friendly_units:
                if unit.type_id == UnitTypeId.DISRUPTOR:
                    disruptor_unit = unit
                    break
                    
            if not disruptor_unit:
                print("DEBUG: No disruptor unit found in friendly units")
                return None
                
            print(f"DEBUG: Using disruptor at position {disruptor_unit.position}")
                
            # Get tactical ground grid to access influence values - use property syntax
            try:
                tactical_grid = self.mediator.get_tactical_ground_grid
                if tactical_grid is None:
                    print("DEBUG: Tactical grid is None - cannot select target")
                    return None
                    
                print(f"DEBUG: Tactical grid shape: {tactical_grid.shape}")
                
                # Debug check for NaN and infinite values
                nan_count = np.isnan(tactical_grid).sum()
                inf_count = np.isinf(tactical_grid).sum()
                print(f"DEBUG: Grid contains {nan_count} NaN values and {inf_count} infinite values")
            except Exception as e:
                print(f"ERROR accessing tactical grid: {str(e)}")
                return None
                
            # Get exclusion mask to avoid targeting the same area multiple times
            exclusion_mask = None
            if nova_manager:
                try:
                    exclusion_mask = nova_manager.get_exclusion_mask(tactical_grid)
                    print(f"DEBUG: Got exclusion mask with {np.sum(exclusion_mask)} cells excluded out of {tactical_grid.size}")
                except Exception as e:
                    print(f"ERROR getting exclusion mask: {e}")
            
            # Helper function to find points within a circle
            def bounded_circle(center, radius, shape):
                xx, yy = np.ogrid[:shape[1], :shape[0]]  # Note: shape[1] is width (x), shape[0] is height (y)
                circle = (xx - center[0])**2 + (yy - center[1])**2
                return np.nonzero(circle <= radius**2)
            
            # Get disruptor position as grid coordinates
            disruptor_pos = disruptor_unit.position.rounded
            pos_x, pos_y = int(disruptor_pos.x), int(disruptor_pos.y)  # x, y for game coords
            
            print(f"DEBUG: Disruptor grid position: ({pos_x}, {pos_y})")
            
            # Find the highest influence area within disruptor range
            try:
                # Get points within radius of disruptor
                points = bounded_circle(center=(pos_x, pos_y), radius=disruptor_max_range, shape=tactical_grid.shape)
                
                if len(points[0]) == 0:
                    print(f"DEBUG: No points found within range {disruptor_max_range} of disruptor")
                    return None
                    
                print(f"DEBUG: Found {len(points[0])} points within range {disruptor_max_range}")
                
                # Get values at those points
                values = tactical_grid[points]
                
                # Create a mask for finite values only
                finite_mask = np.isfinite(values)
                
                # If we have no finite values, we can't target anything
                if np.sum(finite_mask) == 0:
                    print("DEBUG: No finite influence values found in range")
                    return None
                
                # Get only the finite values
                finite_values = values[finite_mask]
                finite_indices = np.where(finite_mask)[0]
                
                # Apply exclusion mask if available
                if exclusion_mask is not None:
                    # Get exclusion status at these points
                    exclusion_status = exclusion_mask[points]
                    # Combined mask: finite values AND not excluded
                    eligible_mask = finite_mask & ~exclusion_status
                    eligible_indices = np.where(eligible_mask)[0]
                    
                    if len(eligible_indices) == 0:
                        print("DEBUG: All points are either infinite or excluded")
                        return None
                        
                    # Only consider eligible values
                    eligible_values = values[eligible_indices]
                    
                    # Find the maximum value and its index
                    if len(eligible_values) == 0:
                        print("DEBUG: No eligible values after applying masks")
                        return None
                        
                    max_value = np.max(eligible_values)
                    max_eligible_index = np.argmax(eligible_values)
                    max_index = eligible_indices[max_eligible_index]
                else:
                    # No exclusion mask, find max among all finite values
                    max_value = np.max(finite_values)
                    max_finite_index = np.argmax(finite_values)
                    max_index = finite_indices[max_finite_index]
                
                # Get the position of the maximum value
                max_x, max_y = points[0][max_index], points[1][max_index]
                
                # Convert the grid position back to a proper game world Point2
                game_world_pos = Point2((max_x, max_y))
                
                # Print positions of enemy units for debugging
                if enemy_units:
                    print(f"DEBUG: Enemy unit positions: {[unit.position for unit in enemy_units]}")
                
                print(f"DEBUG: Found maximum influence value {max_value} at position {game_world_pos}")
                
                # Check if the value is high enough to be worth targeting
                influence_threshold = 100
                if max_value > influence_threshold:
                    # Find enemy units within the disruptor's range that could potentially be hit
                    nearby_enemies = [unit for unit in enemy_units 
                                     if unit.position.distance_to(disruptor_pos) <= disruptor_max_range]
                    
                    # Print debug info about enemy positions
                    enemy_positions = [(unit.position.x, unit.position.y) for unit in nearby_enemies]
                    print(f"DEBUG: Enemy unit positions: {enemy_positions}")
                    
                    if nearby_enemies:
                        try:
                            # Since cy_find_aoe_position isn't available, implement our own AOE targeting
                            # Find the best position to hit the most enemies with the Nova
                            best_position = None
                            max_enemies_hit = 0
                            
                            # Create a list of candidate positions - all enemy unit positions
                            candidate_positions = []
                            for center_unit in nearby_enemies:
                                # Get grid coordinates for this unit
                                center_pos = center_unit.position
                                grid_x, grid_y = int(center_pos.x), int(center_pos.y)
                                
                                # Skip this position if it's in an excluded area or out of bounds
                                if (0 <= grid_x < tactical_grid.shape[0] and 
                                    0 <= grid_y < tactical_grid.shape[1]):
                                    
                                    # Check if this position is excluded
                                    is_excluded = False
                                    if exclusion_mask is not None and exclusion_mask[grid_x, grid_y]:
                                        print(f"DEBUG: Position {center_pos} is in excluded area")
                                        is_excluded = True
                                    
                                    # Even if excluded, add to candidates but mark as excluded
                                    candidate_positions.append((center_unit, is_excluded))
                            
                            # Process candidate positions, prioritizing non-excluded positions
                            # First try non-excluded positions
                            for center_unit, is_excluded in candidate_positions:
                                if is_excluded:
                                    continue  # Skip excluded positions on first pass
                                    
                                # Count enemies within nova_radius of this unit
                                enemies_hit = sum(1 for unit in nearby_enemies 
                                               if unit.position.distance_to(center_unit.position) <= nova_radius)
                                
                                # If this is better than our previous best, update it
                                if enemies_hit > max_enemies_hit:
                                    # Check for friendly fire
                                    friendly_hit = any(unit.position.distance_to(center_unit.position) <= nova_radius 
                                                    for unit in friendly_units)
                                    
                                    # Only use this position if it doesn't hit friendly units
                                    if not friendly_hit:
                                        max_enemies_hit = enemies_hit
                                        best_position = center_unit.position
                            
                            # If we didn't find a good non-excluded position, don't use excluded positions
                            if best_position is None:
                                print("DEBUG: No good non-excluded positions found, skipping Nova to avoid wasteful firing")
                                return None
                            
                            # Only use the position if it's valid and within range
                            if best_position and best_position.distance_to(disruptor_unit.position) <= disruptor_max_range:
                                print(f"DEBUG: Best target at {best_position} will hit {max_enemies_hit} enemy units")
                                
                                return best_position
                            else:
                                print(f"DEBUG: Best position {best_position} is out of disruptor range")
                        except Exception as e:
                            print(f"ERROR in AOE targeting: {e}")
                    else:
                        print(f"DEBUG: No nearby enemy units within disruptor range")
                else:
                    print(f"DEBUG: Maximum influence value {max_value} below threshold {influence_threshold}")
                
                return None
                
            except Exception as e:
                print(f"ERROR during target selection: {e}")
                import traceback
                traceback.print_exc()
                return None
                
        except Exception as e:
            print(f"ERROR in select_best_target: {e}")
            import traceback
            traceback.print_exc()
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
            nearby_enemies = [unit for unit in enemy_units 
                             if unit.position.distance_to(current_position) <= MAX_SEARCH_RADIUS]
            
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
                    print(f"DEBUG: Target {target} rejected by nova_manager - not firing to avoid wasting Nova")
                    return None
                else:
                    print(f"DEBUG: Target registered successfully")
            except Exception as e:
                print(f"ERROR registering target: {e}")
                return None  # Don't fire if we can't register the target

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

    def update_info(self):
        """Update nova tracking state each step."""
        self.frames_left -= 1  # Decrement frame count
        self.distance_left = self.calculate_distance_left(self.unit.movement_speed)
    
    def calculate_distance_left(self, unit_speed: float) -> float:
        """Calculate remaining distance based on unit movement speed and frames left.

        Uses the constant 22.4 frames per second to convert speed into distance per frame.
        """
        return self.frames_left * (unit_speed / 22.4)