"""Custom behavior for Disruptor Nova ability.
Implements targeting logic, firing mechanics, and tracking for the Disruptor's Purification Nova.
Handles target selection based on tactical influence grid, unit positioning, and damage potential.
Integrates with the Ares combat framework for coordinated unit control.
"""
from typing import TYPE_CHECKING, List, Optional
from sc2.data import UnitTypeId
from sc2.ids.ability_id import AbilityId


from behaviors.combat.individual.combat_individual_behavior import CombatIndividualBehavior
from ares.managers.manager_mediator import ManagerMediator
import numpy as np
from sc2.position import Point2

class UseDisruptorNova(CombatIndividualBehavior):
    def __init__(self, mediator: ManagerMediator, bot: 'AresBot', position_update_frequency: int = 10):
        """Initialize the Disruptor Nova behavior controller.
        
        Args:
            mediator: Access point for game state and tactical information
            bot: Reference to the main bot instance
            position_update_frequency: How often to update Nova position (in frames)
        """
        # Nova ability constants
        self.cooldown = 21.4  # Cooldown in seconds
        self.nova_duration = 2.1  # Duration in seconds
        
        # Targeting state
        self.best_target_pos = None
        self.best_target_influence = 0
        self.unit = None
        self.original_target_pos = None
        self.target_pos = None
        
        # Execution state
        self.executing = False
        self.frames_left = 0
        self.distance_left = 0.0
        self.mediator = mediator
        self.bot = bot
        self.should_debug_visuals = False
        
        # Tactical grid reference - retrieved fresh when needed
        self.influence_grid = None
        
        print("DEBUG: UseDisruptorNova initialized")
        
        # Movement control
        self.last_movement_frame = 0
        self.movement_cooldown = 5  # Frames between movement commands
        
        # Execution timing
        self.execution_start_time = 0
        self.max_execution_time = 10.0  # Maximum execution time in seconds
        
        # Position tracking
        self.max_distance_tracking_factor = 35.0  # Max tracking distance
        self.frame_counter = 0
        self.position_update_frequency = position_update_frequency
        self.current_position = None
        self.initial_unit_position = None
        self.last_target_update_time = 0
    
    def can_use(self, disruptor_unit) -> bool:
        """Check if the Disruptor can use Purification Nova.
        
        Args:
            disruptor_unit: The Disruptor unit to check
            
        Returns:
            bool: True if Nova is available, False otherwise
        """
        return (AbilityId.EFFECT_PURIFICATIONNOVA in disruptor_unit.abilities)

    def select_best_target(self, enemy_units: List['Unit'], friendly_units: List['Unit'], nova_manager) -> Optional[Point2]:
        """
        Find optimal Nova target position based on tactical influence grid.
        
        Evaluates potential targets using:
        1. Tactical influence grid for high-value areas
        2. Exclusion zones to prevent overlapping Novas
        
        Args:
            enemy_units: List of enemy units to consider targeting
            friendly_units: List of friendly units to avoid damaging
            nova_manager: NovaManager for target coordination and exclusion zones
            
        Returns:
            Point2: Optimal target position, or None if no suitable target found
        """
        try:
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
                    
                    # Find indices of non-excluded points
                    non_excluded_mask = ~exclusion_status[finite_mask]
                    
                    # If we have non-excluded points, use only those
                    if np.sum(non_excluded_mask) > 0:
                        finite_values = finite_values[non_excluded_mask]
                        finite_indices = finite_indices[non_excluded_mask]
                    else:
                        print("DEBUG: All points are excluded, no valid targets")
                        return None
                
                # Find the index of the maximum value
                max_index = np.argmax(finite_values)
                max_value = finite_values[max_index]
                
                # Convert back to original indices
                original_index = finite_indices[max_index]
                
                # Get the position of the maximum value
                max_x, max_y = points[0][original_index], points[1][original_index]
                
                # Convert the grid position back to a proper game world Point2
                game_world_pos = Point2((max_x, max_y))
                
                # Print positions of enemy units for debugging
                if enemy_units:
                    print(f"DEBUG: Enemy unit positions: {[unit.position for unit in enemy_units]}")
                
                print(f"DEBUG: Found maximum influence value {max_value} at position {game_world_pos}")
                
                # Check if the value is high enough to be worth targeting
                influence_threshold = 200
                if max_value > influence_threshold:
                    # Count how many enemies would be hit by this position
                    nearby_enemies = [unit for unit in enemy_units 
                                      if unit.position.distance_to(game_world_pos) <= nova_manager.nova_radius]
                    
                    if nearby_enemies:
                        print(f"DEBUG: Best target at {game_world_pos} will hit {len(nearby_enemies)} enemy units")
                        return game_world_pos
                    else:
                        print(f"DEBUG: No enemy units would be hit at position {game_world_pos}")
                        return None
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
        Dynamically adjust Nova targeting during flight based on battlefield changes.
        
        Evaluates if a more optimal target has become available within the Nova's
        remaining travel range and updates targeting accordingly.
        
        Args:
            enemy_units: List of enemy units to consider
            friendly_units: List of friendly units to avoid
            nova_manager: NovaManager for exclusion zones and target registration
            
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
        """Initialize Nova tracking state when a Nova is fired.
        
        Args:
            unit: The Nova unit to track
        """
        self.unit = unit
        self.frames_left = 48  # Nova lifetime (2.1s * 22.4 fps)
        self.distance_left = self.calculate_distance_left(unit.movement_speed)
        self.best_target_pos = unit.position  # Initial target position
        

    def execute(self, disruptor_unit, enemy_units: List['Unit'], friendly_units: List['Unit'], nova_manager=None):
        """Fire a Disruptor's Purification Nova at the best available target.
        
        Handles target selection, firing mechanics, and Nova registration with the manager.
        Includes range checking and positioning logic to ensure effective Nova usage.
        
        Args:
            disruptor_unit: The Disruptor unit to fire the Nova
            enemy_units: List of enemy units to consider targeting
            friendly_units: List of friendly units to avoid damaging
            nova_manager: Optional NovaManager for target coordination
            
        Returns:
            UseDisruptorNova: This instance if Nova fired successfully, None otherwise
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
        """Process one game step for an active Nova.
        
        Updates Nova state, decrements lifetime counter, and handles movement commands.
        Dynamically adjusts targeting based on battlefield changes when appropriate.
        
        Args:
            enemy_units: List of enemy units for targeting updates
            friendly_units: List of friendly units to avoid
            nova_manager: Optional NovaManager for coordinated targeting
            
        Returns:
            bool: True if Nova is still active, False if expired
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
            self.unit.move(self.best_target_pos)
            print(f"Moving Nova to target: {self.best_target_pos}")
        
        # Optionally, output debug information
        # Continue running until the nova expires
        return self.frames_left > 0

    def update_info(self):
        """Update Nova tracking state for the current game step.
        
        Decrements frame counter and recalculates remaining travel distance.
        """
        self.frames_left -= 1
        self.distance_left = self.calculate_distance_left(self.unit.movement_speed)
    
    def calculate_distance_left(self, unit_speed: float) -> float:
        """Calculate maximum remaining travel distance for the Nova.
        
        Converts unit speed (game units per second) to distance per frame,
        then multiplies by remaining frames to get total possible distance.
        
        Args:
            unit_speed: Movement speed of the Nova in game units per second
            
        Returns:
            float: Maximum remaining travel distance in game units
        """
        return self.frames_left * (unit_speed / 22.4)