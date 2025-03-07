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

from cython_extensions import cy_all_points_below_max_value


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

    def select_best_target(self, disruptor_unit, enemy_units: List['Unit'], friendly_units: List['Unit'], nova_manager=None):
        """
        Select the best target position for the nova.
        Uses the tactical grid to find optimal target areas.
        
        Args:
            disruptor_unit: The Disruptor unit that will fire the Nova
            enemy_units: List of enemy units to consider as targets
            friendly_units: List of friendly units to avoid damaging
            nova_manager: Optional NovaManager for target coordination
            
        Returns:
            The Point2 target position or None if no valid target found
        """
        if not enemy_units:
            print("DEBUG select_best_target: No enemy units to target")
            return None
            
        try:
            # Define the maximum search radius to limit computational load
            MAX_SEARCH_RADIUS = 15.0  # Units beyond this distance won't be considered
            
            # Filter units by distance to the disruptor to improve performance
            if disruptor_unit:
                disruptor_pos = disruptor_unit.position
                nearby_enemies = [unit for unit in enemy_units if unit.position.distance_to(disruptor_pos) <= MAX_SEARCH_RADIUS]
                
                print(f"DEBUG select_best_target: Filtered {len(nearby_enemies)} nearby enemies out of {len(enemy_units)} total")
                
                if not nearby_enemies:
                    print("DEBUG select_best_target: No nearby enemy units to target")
                    return None
            else:
                # If no disruptor unit is provided, use all units (less efficient)
                nearby_enemies = enemy_units
            
            # Get the tactical grid from the mediator
            grid = self.mediator.get_tactical_ground_grid
            
            if grid is None:
                print("DEBUG select_best_target: Tactical grid is None")
                return None
                
            # Get exclusion mask for other active Nova targets
            exclusion_mask = None
            try:
                if nova_manager:
                    exclusion_mask = nova_manager.get_exclusion_mask(grid)
            except Exception as e:
                print(f"DEBUG ERROR getting exclusion mask: {e}")
            
            # Find the position with the highest influence value
            best_pos = None
            best_influence = float('-inf')
            
            for enemy in nearby_enemies:
                try:
                    # Check the enemy position
                    enemy_pos = enemy.position
                    print(f"DEBUG: Checking enemy at {enemy_pos}")
                    
                    # Check grid value at enemy position
                    try:
                        # Convert game position to grid indices
                        grid_x = int(enemy_pos.x)
                        grid_y = int(enemy_pos.y)
                        
                        # Skip if this position is in an exclusion zone
                        if exclusion_mask is not None:
                            if exclusion_mask[grid_y, grid_x]:
                                print(f"DEBUG: Enemy at {enemy_pos} is in an exclusion zone - skipping")
                                continue
                        
                        # Get influence value at this position
                        influence = grid[grid_y, grid_x]
                        print(f"DEBUG: Enemy at {enemy_pos} has influence {influence}")
                        
                        # Use a very low threshold to accept almost any target
                        # We'd rather have some target than none
                        if influence > best_influence:
                            best_influence = influence
                            best_pos = enemy_pos
                            print(f"DEBUG: Found better position at {best_pos} with influence {best_influence}")
                    except Exception as e:
                        print(f"DEBUG ERROR getting influence value: {e}")
                except Exception as e:
                    print(f"DEBUG ERROR processing enemy unit: {e}")
            
            # If we found a good position, return it
            if best_pos is not None:
                # Store for future comparisons
                self.best_target_pos = best_pos
                self.best_target_influence = best_influence
                
                print(f"DEBUG select_best_target: Found target at {best_pos} with influence {best_influence}")
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
            # Get the tactical grid
            grid = self.mediator.get_tactical_ground_grid
            if grid is None:
                print("DEBUG: Tactical grid is None in update_target_position")
                return False
            
            # Calculate maximum distance the Nova can still travel based on remaining frames
            if nova_manager and hasattr(nova_manager, 'nova_speed'):
                # Calculate max travel distance based on remaining frames
                frames_per_second = 22.4  # SC2 runs at 22.4 frames per second
                remaining_seconds = self.frames_left / frames_per_second
                max_travel_distance = nova_manager.nova_speed * remaining_seconds
                
                # Current Nova position is our starting point
                current_position = self.unit.position
                
                print(f"DEBUG update_target: {self.frames_left} frames left, max travel distance: {max_travel_distance:.2f}")
                
                # Filter enemy units by distance to the Nova unit
                MAX_SEARCH_RADIUS = max_travel_distance + 2.0  # Add a small buffer
                nearby_enemies = [unit for unit in enemy_units if unit.position.distance_to(current_position) <= MAX_SEARCH_RADIUS]
                
                if not nearby_enemies:
                    print("DEBUG update_target: No nearby enemy units within reach")
                    return False
                
                # Get the exclusion mask for other active Nova targets
                exclusion_mask = None
                try:
                    if nova_manager:
                        # Get exclusion mask but ignore our current target
                        exclusion_mask = nova_manager.get_exclusion_mask(grid)
                        
                        # Modify the mask to not exclude our current target
                        if exclusion_mask is not None and self.best_target_pos:
                            # Convert target position to grid indices
                            target_x, target_y = int(self.best_target_pos.x), int(self.best_target_pos.y)
                            
                            # Create a small area around our current target that's allowed
                            if 0 <= target_y < exclusion_mask.shape[0] and 0 <= target_x < exclusion_mask.shape[1]:
                                radius = int(nova_manager.exclusion_radius)
                                y_min = max(0, target_y - radius)
                                y_max = min(exclusion_mask.shape[0], target_y + radius + 1)
                                x_min = max(0, target_x - radius)
                                x_max = min(exclusion_mask.shape[1], target_x + radius + 1)
                                
                                # Allow our current target area
                                exclusion_mask[y_min:y_max, x_min:x_max] = False
                except Exception as e:
                    print(f"DEBUG ERROR getting exclusion mask in update: {e}")
                
                # Find the position with the highest influence value among nearby enemies
                best_pos = None
                best_influence = float('-inf')
                
                for enemy in nearby_enemies:
                    try:
                        # Check the enemy position
                        enemy_pos = enemy.position
                        
                        # Check if this position is reachable
                        if not nova_manager.can_nova_reach_target(current_position, enemy_pos, self.frames_left):
                            continue
                        
                        # Check grid value at enemy position
                        try:
                            # Convert game position to grid indices
                            grid_x = int(enemy_pos.x)
                            grid_y = int(enemy_pos.y)
                            
                            # Skip if this position is in an exclusion zone
                            if exclusion_mask is not None:
                                if exclusion_mask[grid_y, grid_x]:
                                    continue
                            
                            # Get influence value at this position
                            influence = grid[grid_y, grid_x]
                            
                            # Use a very low threshold to accept almost any target
                            # We'd rather have some target than none
                            if influence > best_influence:
                                best_influence = influence
                                best_pos = enemy_pos
                                print(f"DEBUG update_target: Found better position at {best_pos} with influence {best_influence}")
                        except Exception as e:
                            print(f"DEBUG ERROR getting influence value: {e}")
                    except Exception as e:
                        print(f"DEBUG ERROR processing enemy unit: {e}")
                
                # If we found a good position, return it
                if best_pos is not None:
                    # Only update if the new influence is significantly better or the position is significantly different
                    influence_improvement_threshold = 50.0
                    position_change_threshold = 3.0
                    
                    should_update = False
                    
                    # Update if the new influence is significantly better
                    if best_influence > self.best_target_influence + influence_improvement_threshold:
                        should_update = True
                        print(f"DEBUG update_target: New influence {best_influence:.1f} is better than current {self.best_target_influence:.1f}")
                    
                    # Or update if the position is significantly different
                    elif self.best_target_pos.distance_to(best_pos) > position_change_threshold:
                        should_update = True
                        print(f"DEBUG update_target: New position {best_pos} is significantly different from current {self.best_target_pos}")
                    
                    if should_update:
                        # Unregister old target
                        if nova_manager:
                            nova_manager.unregister_nova_target(self.best_target_pos)
                            
                        # Update target
                        old_pos = self.best_target_pos
                        self.best_target_pos = best_pos
                        self.best_target_influence = best_influence
                        
                        # Register new target
                        if nova_manager:
                            success = nova_manager.register_nova_target(best_pos)
                            print(f"DEBUG update_target: {'Successfully' if success else 'Failed to'} register new target")
                        
                        print(f"DEBUG update_target: Updated target from {old_pos} to {best_pos}")
                        return True
                
                return False
                
        except Exception as e:
            print(f"DEBUG ERROR in update_target_position: {e}")
            return False

    def load_info(self, nova_unit) -> None:
        """Load information about the nova unit."""
        self.unit = nova_unit
        self.current_position = nova_unit.position
        self.frames_left = round(2.1 * 22.4)  # ~2.1 seconds at 22.4 frames per second
        self.executing = True
        
        # Initialize movement tracking
        self.previous_position = nova_unit.position
        self.consecutive_stuck_frames = 0
        
        # Reset the frame counter
        self.frame_counter = 0
        
        # Initialize time-based tracking
        import time
        self.execution_start_time = time.time()
        self.last_position_update_time = time.time()
        print(f"DEBUG: Loaded Nova unit - initial position: {self.current_position}")

    def execute(self, disruptor_unit, enemy_units, friendly_units=None, nova_manager=None) -> bool:
        """
        Execute nova ability on the disruptor unit.
        
        Args:
            disruptor_unit: The disruptor unit to use the nova with
            enemy_units: List of enemy units to consider as targets
            friendly_units: (Optional) List of friendly units to avoid damaging
            nova_manager: (Optional) The NovaManager instance for coordination
            
        Returns:
            bool: True if nova was successfully executed or is still active, False otherwise
        """
        import time
        current_time = time.time()
        
        # Case 1: Nova is already executing - check if it's still valid
        if self.executing:
            # Time-based safety check: Reset if this Nova has been executing for too long
            if current_time - self.execution_start_time > self.max_execution_time:
                print(f"DEBUG: Nova has exceeded maximum execution time, resetting")
                # Unregister the target position if we have a manager
                if nova_manager and self.best_target_pos:
                    try:
                        nova_manager.unregister_nova_target(self.best_target_pos)
                    except Exception as e:
                        print(f"DEBUG ERROR unregistering target during reset: {e}")
                # Reset the Nova state
                self.reset()
                # Return False to indicate Nova is not active
                return False
            
            # Verify the Nova unit still exists
            if self.unit is None or not hasattr(self.unit, 'is_alive') or not self.unit.is_alive:
                print(f"DEBUG: Nova unit is no longer valid, resetting")
                # Unregister the target position if we have a manager
                if nova_manager and self.best_target_pos:
                    try:
                        nova_manager.unregister_nova_target(self.best_target_pos)
                    except Exception as e:
                        print(f"DEBUG ERROR unregistering target during reset: {e}")
                # Reset the Nova state
                self.reset()
                # Return False to indicate Nova is not active
                return False
                
            # Nova is already executing and still valid - continue execution
            print(f"DEBUG: Nova already executing, frames left: {self.frames_left}")
            return True  # Return True to indicate Nova is still active

        # Case 2: Nova is not executing - start new execution
        try:
            # Find the best target position using enemy_units
            target_pos = self.select_best_target_pos(disruptor_unit, enemy_units, friendly_units, nova_manager)
            
            # If no valid target found, return False
            if not target_pos:
                print(f"DEBUG: No valid target position found for Nova")
                return False
                
            # Save the original target position
            self.original_target_pos = target_pos
            # Initialize best_target_pos to original position
            self.best_target_pos = target_pos
            
            # Register the target position with the Nova manager
            if nova_manager:
                registration_success = nova_manager.register_nova_target(target_pos)
                if not registration_success:
                    print(f"DEBUG: Could not register Nova target at {target_pos}, existing target at this position")
                    return False
                print(f"DEBUG: Successfully registered Nova target at {target_pos}")
            
            # Try to execute Nova ability
            ability_executed = self.bot.do(disruptor_unit(AbilityId.EFFECT_PURIFICATIONNOVA, target_pos))
            
            if ability_executed:
                print(f"DEBUG: Nova ability executed successfully at {target_pos}")
                self.load_info(disruptor_unit)
                self.initial_unit_position = disruptor_unit.position
                # We'll set the current target to the launch position
                self.best_target_pos = target_pos
                return True
            else:
                print(f"DEBUG: Nova ability execution failed")
                # If execution failed, unregister the target
                if nova_manager and self.best_target_pos:
                    nova_manager.unregister_nova_target(self.best_target_pos)
                return False
        except Exception as e:
            print(f"DEBUG ERROR executing Nova: {e}")
            # If error during execution, ensure the target is unregistered
            if nova_manager and self.best_target_pos:
                nova_manager.unregister_nova_target(self.best_target_pos)
            return False
    
    def select_best_target_pos(self, disruptor_unit, enemy_units, friendly_units=None, nova_manager=None) -> Point2:
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

    def update_info(self) -> bool:
        """
        Update nova information.
        
        Returns:
            bool: True if successfully updated, False if Nova is no longer valid
        """
        try:
            # Increment our frame counter (used for debugging and movement logic)
            self.frame_counter += 1
            
            # Reduce remaining frames
            self.frames_left -= 1
            
            # If Nova unit is invalid or no frames left, unit is done
            if self.unit is None or self.frames_left <= 0:
                self.executing = False
                return False
                
            # If unit is valid, update position information
            if hasattr(self.unit, 'position'):
                try:
                    # Update current position
                    new_position = self.unit.position
                    
                    # If the position changed, update our tracking
                    if self.current_position is not None and self.current_position != new_position:
                        # Store previous position for movement detection
                        if not hasattr(self, 'previous_position'):
                            self.previous_position = self.current_position
                        else:
                            # Only update previous_position if significant time has passed
                            import time
                            current_time = time.time()
                            if not hasattr(self, 'last_position_update_time'):
                                self.last_position_update_time = current_time
                                self.previous_position = self.current_position
                            elif current_time - self.last_position_update_time > 0.1:  # Update every 100ms
                                self.previous_position = self.current_position
                                self.last_position_update_time = current_time
                    
                    # Update current position to new value
                    self.current_position = new_position
                    
                    # Update target periodically if we have access to the necessary data
                    if hasattr(self, 'last_target_update_time'):
                        import time
                        current_time = time.time()
                        if current_time - self.last_target_update_time > 0.5:  # Every 0.5 seconds
                            self.last_target_update_time = current_time
                            # Note: actual target update happens in run_step
                    
                    # Successfully updated info
                    return True
                except (AttributeError, Exception) as e:
                    # If we can't access the position, the unit is invalid
                    print(f"DEBUG: Nova unit is invalid or expired: {e}")
                    self.frames_left = 0
                    self.executing = False
                    return False
            else:
                # Nova unit is no longer valid
                self.frames_left = 0
                self.executing = False
                print(f"DEBUG: Nova unit is None or has no position, marking as not executing")
                return False
        except Exception as e:
            print(f"DEBUG ERROR in update_info: {e}")
            # If error in updating info, mark as done
            self.frames_left = 0
            self.executing = False
            return False
    
    def run_step(self, enemy_units, friendly_units, nova_manager) -> bool:
        """
        Run one step of the nova logic, controlling its movement towards enemies.
        
        Returns:
            bool: True if Nova is still active, False otherwise
        """
        try:
            # First update unit info to get fresh position data
            updated = self.update_info()
            if not updated:
                # Failed to update info, probably unit is gone
                if self.best_target_pos and nova_manager:
                    print(f"DEBUG: Nova unit info update failed, unregistering target")
                    nova_manager.unregister_nova_target(self.best_target_pos)
                
                # Reset Nova state
                self.reset()
                return False
                
            # Check if frames are done
            if self.frames_left <= 0:
                # Time's up, unregister target and reset
                if self.best_target_pos and nova_manager:
                    print(f"DEBUG: Nova frames completed, unregistering target")
                    nova_manager.unregister_nova_target(self.best_target_pos)
                
                # Reset Nova state
                self.reset()
                return False
                
            # Unit should be executing at this point
            if not self.executing:
                print(f"DEBUG: Nova is not in executing state but run_step was called")
                return False
            
            # Check if the Nova has moved since last update
            if hasattr(self, 'previous_position') and self.current_position is not None and self.previous_position is not None:
                movement_distance = self.current_position.distance_to(self.previous_position)
                
                # If we haven't moved significantly in several consecutive frames, we might be stuck
                if movement_distance < 0.1:  # Minimal movement threshold
                    if hasattr(self, 'consecutive_stuck_frames'):
                        self.consecutive_stuck_frames += 1
                        
                        # If we've been stuck for too many frames, try to update the target
                        if self.consecutive_stuck_frames > 10:  # Adjust threshold as needed
                            print(f"DEBUG: Nova appears to be stuck for {self.consecutive_stuck_frames} frames, trying to find better path")
                            # Try to update the target to break out of the stuck state
                            self.update_target_position(enemy_units, friendly_units, nova_manager)
                            # If we still have the same target, simply reset stuck counter and try again
                            self.consecutive_stuck_frames = 0
                else:
                    # We're moving, reset stuck counter
                    if hasattr(self, 'consecutive_stuck_frames'):
                        self.consecutive_stuck_frames = 0
            
            # Store current position for next frame comparison
            if self.current_position is not None:
                self.previous_position = self.current_position
                
            # Periodically try to update target based on changing battlefield
            if self.frame_counter % 10 == 0:  # Check every 10 frames
                target_updated = self.update_target_position(enemy_units, friendly_units, nova_manager)
                if target_updated:
                    print(f"DEBUG: Updated Nova target based on battlefield changes")
            
            # If we have a target position, move the Nova toward it
            if self.best_target_pos:
                # Calculate vector from current position to target
                dx = self.best_target_pos.x - self.current_position.x
                dy = self.best_target_pos.y - self.current_position.y
                
                # Calculate the distance to the target
                distance = (dx**2 + dy**2)**0.5
                
                # Only move if we're not very close to the target
                if distance > 1.0:
                    # Normalize the vector to get direction
                    if distance > 0:
                        dx /= distance
                        dy /= distance
                    
                    # Calculate how far the Nova can move this frame
                    # Nova speed is 5.95 units/sec, divide by 22.4 frames/sec
                    max_move_dist = nova_manager.nova_speed / 22.4
                    
                    # If we're very close to the target, don't overshoot
                    move_dist = min(max_move_dist, distance)
                    
                    # Calculate the new position
                    new_x = self.current_position.x + dx * move_dist
                    new_y = self.current_position.y + dy * move_dist
                    
                    # Create a new Point2 target position
                    new_pos = Point2((new_x, new_y))
                    
                    # Only log position occasionally to reduce spam
                    if self.frame_counter % self.position_update_frequency == 0:
                        print(f"DEBUG: Moving Nova from {self.current_position} toward {self.best_target_pos}, distance: {distance:.2f}")
                    
                    # If a valid target was found and it differs from the current position, command the nova to move
                    if self.best_target_pos is not None and self.unit is not None and hasattr(self.unit, 'move'):
                        try:
                            # Issue the move command directly to the unit
                            self.unit.move(self.best_target_pos)
                            if self.frame_counter % self.position_update_frequency == 0:
                                print(f"Moving Nova to target: {self.best_target_pos}")
                        except Exception as e:
                            print(f"ERROR: Failed to move Nova: {e}")
            
            # Nova is still active
            return True
            
        except Exception as e:
            print(f"DEBUG ERROR in run_step: {e}")
            
            # On error, assume Nova is no longer valid
            # Unregister the target if we have one
            if self.best_target_pos and nova_manager:
                nova_manager.unregister_nova_target(self.best_target_pos)
            
            # Reset the Nova state
            self.reset()
            
            # Nova is no longer active
            return False
    
    def reset(self) -> None:
        """Reset nova state."""
        print(f"DEBUG: Resetting Nova - was executing: {self.executing}, frames left: {self.frames_left}")
        # Reset execution state
        self.executing = False
        self.frames_left = 0
        
        # Clear unit reference
        self.unit = None
        
        # Clear target positions
        self.best_target_pos = None
        self.original_target_pos = None
        
        # Reset tracking variables
        self.frame_counter = 0
        self.current_position = None
        self.previous_position = None
        self.initial_unit_position = None
        self.consecutive_stuck_frames = 0
        
        # Reset timers
        self.execution_start_time = 0
        self.last_position_update_time = 0

    def run_debug(self):
        """Output debugging information."""
        print(f"Nova Frames Left: {self.frames_left}, Distance Left: {self.distance_left}")
        print(f"Current Position: {self.unit.position if hasattr(self.unit, 'position') else self.last_known_position}, Target Position: {self.best_target_pos}")
        
        # For debugging, let's get a fresh grid if we don't have one stored
        self.influence_grid = self.mediator.get_tactical_ground_grid
        
        try:
            # Get max and min values to understand the range of influence
            max_value = np.max(self.influence_grid)
            min_value = np.min(self.influence_grid)
            
            # Calculate percentiles for better understanding of distribution
            valid_grid = self.influence_grid[~np.isinf(self.influence_grid)]
            if len(valid_grid) > 0:
                p10 = np.percentile(valid_grid, 10)
                p25 = np.percentile(valid_grid, 25)
                p50 = np.percentile(valid_grid, 50)  # median
                p75 = np.percentile(valid_grid, 75)
                p90 = np.percentile(valid_grid, 90)
                
                # Count values in different ranges
                below150 = np.sum(valid_grid < 150)
                range150_190 = np.sum((valid_grid >= 150) & (valid_grid < 190))
                range190_210 = np.sum((valid_grid >= 190) & (valid_grid <= 210))
                range210_250 = np.sum((valid_grid > 210) & (valid_grid <= 250))
                above250 = np.sum(valid_grid > 250)
                
                print(f"Grid range - Max: {max_value}, Min: {min_value}, Neutral: 200")
                print(f"Grid percentiles - 10%: {p10:.1f}, 25%: {p25:.1f}, 50%: {p50:.1f}, 75%: {p75:.1f}, 90%: {p90:.1f}")
                print(f"Value distribution - Below 150: {below150}, 150-190: {range150_190}, 190-210: {range190_210}, 210-250: {range210_250}, Above 250: {above250}")
            else:
                print(f"Grid range - Max: {max_value}, Min: {min_value}, Neutral: 200")
                print("No valid (non-infinite) values in grid for percentile analysis")
            
            # Create enemy influence visualization (values > 200)
            enemy_grid = self.influence_grid.copy()
            enemy_grid[enemy_grid <= 200] = 0  # Zero out non-enemy areas
            enemy_grid[enemy_grid > 200] -= 200  # Normalize to positive values starting from 0
            
            # Create friendly influence visualization (values < 200)
            friendly_grid = self.influence_grid.copy()
            friendly_grid[friendly_grid >= 200] = 0  # Zero out non-friendly areas
            friendly_grid = 200 - friendly_grid  # Invert for visualization (make small values large)
            friendly_grid[friendly_grid <= 0] = 0  # Remove any negative values
            
            print(f"Enemy influence - Max: {np.max(enemy_grid) if np.max(enemy_grid) > 0 else 0}")
            print(f"Friendly influence - Max: {np.max(friendly_grid) if np.max(friendly_grid) > 0 else 0}")
            
            # Draw enemy influence (green)
            if np.max(enemy_grid) > 0:
                self.bot.map_data.draw_influence_in_game(
                    grid=enemy_grid,
                    lower_threshold=10,  # Show significant enemy influence
                    upper_threshold=50,  # Cap for better visualization
                    color=(0, 255, 0)    # Green for enemy influence
                )
            
            # Draw friendly influence (red)
            if np.max(friendly_grid) > 0:
                self.bot.map_data.draw_influence_in_game(
                    grid=friendly_grid,
                    lower_threshold=10,   # Show significant friendly influence
                    upper_threshold=50,   # Cap for better visualization
                    color=(255, 0, 0)     # Red for friendly influence
                )
                
        except Exception as e:
            print(f"Error in grid visualization: {e}")

    def _visualize_grid(self, grid: np.ndarray) -> None:
        """
        Debug visualization method to show grid influence values.
        
        Args:
            grid: The tactical ground grid
        """
        try:
            print(f"DEBUG _visualize_grid: Visualizing grid with shape {grid.shape}")
            
            # Store the grid for visualization
            self.influence_grid = grid.copy()
            
            # Get max and min values to understand the range of influence
            max_value = np.max(grid[~np.isinf(grid)])
            min_value = np.min(grid[~np.isinf(grid)])
            print(f"Grid range - Max: {max_value}, Min: {min_value}, Neutral: 200")
            
            # Create enemy influence visualization (values > 200)
            enemy_grid = grid.copy()
            enemy_grid[np.isinf(enemy_grid)] = 0
            enemy_grid[enemy_grid <= 200] = 0  # Zero out non-enemy areas
            enemy_grid[enemy_grid > 200] -= 200  # Normalize to positive values starting from 0
            
            # Create friendly influence visualization (values < 200)
            friendly_grid = grid.copy()
            friendly_grid[np.isinf(friendly_grid)] = 0
            friendly_grid[friendly_grid >= 200] = 0  # Zero out non-friendly areas
            friendly_grid = 200 - friendly_grid  # Invert for visualization (make small values large)
            friendly_grid[friendly_grid <= 0] = 0  # Remove any negative values
            
            print(f"Enemy influence - Max: {np.max(enemy_grid) if np.max(enemy_grid) > 0 else 0}")
            print(f"Friendly influence - Max: {np.max(friendly_grid) if np.max(friendly_grid) > 0 else 0}")
            
            # Draw enemy influence (green)
            if np.max(enemy_grid) > 0:
                self.bot.map_data.draw_influence_in_game(
                    grid=enemy_grid,
                    lower_threshold=10,  # Show significant enemy influence
                    upper_threshold=50,  # Cap for better visualization
                    color=(0, 255, 0)    # Green for enemy influence
                )
            
            # Draw friendly influence (red)
            if np.max(friendly_grid) > 0:
                self.bot.map_data.draw_influence_in_game(
                    grid=friendly_grid,
                    lower_threshold=10,   # Show significant friendly influence
                    upper_threshold=50,   # Cap for better visualization
                    color=(255, 0, 0)     # Red for friendly influence
                )
                
            print(f"DEBUG _visualize_grid: Visualization completed")
                
        except Exception as e:
            print(f"DEBUG ERROR in _visualize_grid: {e}")
