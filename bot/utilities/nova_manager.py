from typing import List, Dict, TYPE_CHECKING, Optional, Set, Tuple
from sc2.ids.unit_typeid import UnitTypeId
import time
import numpy as np
from sc2.position import Point2
import traceback

if TYPE_CHECKING:
    from bot.utilities.use_disruptor_nova import UseDisruptorNova


class NovaManager:
    """Coordinates and manages Disruptor Nova abilities across multiple units.
    
    Handles tracking of active Novas, target registration, exclusion zones,
    and Nova lifetime management to ensure efficient usage of this powerful ability.
    """

    def __init__(self, bot, mediator, debug_output: bool = True):
        """Initialize the Nova Manager.
        
        Args:
            bot: Reference to the main bot instance
            mediator: Access point for game state and tactical information
        """
        # Core references
        self.bot = bot
        self.mediator = mediator
        self.debug_output = debug_output
        
        # Nova tracking
        self.active_novas: List = []
        self.current_targets: Dict[str, Point2] = {}
        self.pending_targets: Dict[str, Point2] = {}
        
        # Nova constants
        self.nova_speed = 4.25  # Movement speed (game units/second)
        self.nova_lifetime = 2.1  # Lifetime (seconds)
        self.nova_radius = 1.5  # Explosion radius
        
        # Targeting parameters
        self.exclusion_radius = 3.0  # Exclusion radius in game units
        
        # Unit tracking
        self.enemy_units = []
        self.friendly_units = []
        self.current_time = time.time()

        if self.debug_output:
            print(f"DEBUG: NovaManager initialized with exclusion radius of {self.exclusion_radius} game units")

        try:
            grid = self.mediator.get_tactical_ground_grid
            if self.debug_output:
                if grid is not None:
                    print(f"DEBUG: Successfully accessed tactical grid")
                else:
                    print("DEBUG: Warning - tactical grid is None")
        except Exception as e:
            print(f"DEBUG ERROR initializing NovaManager - could not access tactical grid: {e}")

    def add_nova(self, nova) -> None:
        """Register a nova unit or behavior instance for tracking.
        
        Handles both raw Nova units and Nova behavior instances by wrapping
        raw units in a behavior controller if needed.
        
        Args:
            nova: Either a Nova unit or a UseDisruptorNova behavior instance
        """
        from bot.utilities.use_disruptor_nova import UseDisruptorNova
        self.nova_speed = nova.movement_speed
        if self.debug_output:
            print(f"DEBUG: NovaManager movement speed set to {self.nova_speed} game units/second")

        if not hasattr(nova, 'update_info'):
            nova_instance = UseDisruptorNova(mediator=self.mediator, bot=self.bot)
            nova_instance.load_info(nova)
            self.active_novas.append(nova_instance)
        else:
            self.active_novas.append(nova)
            
        # Register the target position for this nova
        if hasattr(nova, 'best_target_pos') and nova.best_target_pos:
            self.register_nova_target(nova.best_target_pos)

    
    def add_pending_target(self, position: Point2) -> str:
        """
        Register a position as a pending Nova target.
        
        Pending targets are included in exclusion zones but aren't yet confirmed.
        This prevents multiple Disruptors from choosing the same target in a single game step.
        
        Args:
            position: Target position in game coordinates
            
        Returns:
            str: Unique ID for this pending target
        """
        # Generate a unique ID for this pending target
        target_id = f"pending_{time.time()}_{position.x}_{position.y}"
        self.pending_targets[target_id] = position
        if self.debug_output:
            print(f"DEBUG: Added pending target at {position} with ID {target_id}")
        return target_id
    
    def confirm_pending_target(self, target_id: str) -> None:
        """
        Confirm a pending target, moving it to active targets.
        
        Args:
            target_id: The ID of the pending target to confirm
        """
        if target_id in self.pending_targets:
            position = self.pending_targets[target_id]
            # Generate new target ID with proper prefix for active targets
            new_target_id = f"target_{time.time()}_{position.x}_{position.y}"
            self.current_targets[new_target_id] = position
            del self.pending_targets[target_id]
            if self.debug_output:
                print(f"DEBUG: Confirmed pending target {target_id} at {position}, registered as active with ID {new_target_id}")
    
    def cancel_pending_target(self, target_id: str) -> None:
        """
        Cancel a pending target that wasn't used.
        
        Args:
            target_id: The ID of the pending target to cancel
        """
        if target_id in self.pending_targets:
            if self.debug_output:
                print(f"DEBUG: Canceled pending target {target_id} at {self.pending_targets[target_id]}")
            del self.pending_targets[target_id]
    
    def update(self, enemy_units: List['Unit'], friendly_units: List['Unit']) -> None:
        """
        Process one game step for all active Novas.
        
        Updates Nova states, removes expired Novas, and maintains target registrations.
        Called once per game step to manage the lifecycle of all active Novas.
        
        Args:
            enemy_units: Current list of visible enemy units
            friendly_units: Current list of friendly units
        """
        try:
            # Update internal state
            self.enemy_units = enemy_units
            self.friendly_units = friendly_units
            self.current_time = time.time()  # Just for generating unique keys
            
            # Perform periodic cleanup of expired targets
            self.clean_expired_targets()
            
            # Track and update active nova instances
            expired_novas = []
            
            for nova in self.active_novas:
                # Decrement the frame counter and update remaining distance
                nova.update_info()
                
                # If still active, process targeting updates, otherwise remove it
                is_active = nova.run_step(enemy_units, friendly_units, self)
                
                if not is_active:
                    expired_novas.append(nova)
                    if hasattr(nova, 'best_target_pos') and nova.best_target_pos:
                        if self.debug_output:
                            print(f"DEBUG: Unregistering expired Nova target at {nova.best_target_pos}")
                        self.unregister_nova_target(nova.best_target_pos)
                        
            # Remove expired novas
            for expired_nova in expired_novas:
                if expired_nova in self.active_novas:
                    self.active_novas.remove(expired_nova)
                if self.debug_output:
                    print(f"DEBUG: Removed expired nova, {len(self.active_novas)} novas remaining")
                
            # Debug current target state
            if self.debug_output and self.current_targets:
                print(f"DEBUG: Current active Nova targets: {len(self.current_targets)}")
                for key, target in list(self.current_targets.items())[:3]:  # Show max 3 to avoid spam
                    print(f"DEBUG: Active target: {target} (key: {key})")
                if len(self.current_targets) > 3:
                    print(f"DEBUG: ...and {len(self.current_targets)-3} more")
        except Exception as e:
            print(f"DEBUG ERROR in NovaManager.update(): {e}")
            traceback.print_exc()

    def get_active_novas(self):
        """Get all currently active Nova instances.
        
        Returns:
            List: Active UseDisruptorNova behavior instances
        """
        return self.active_novas
        
    def register_nova_target(self, position: Point2, source_pending_id: Optional[str] = None) -> bool:
        """
        Register a position as a Nova target to prevent overlapping Novas.
        
        Maintains a registry of active Nova targets and enforces minimum
        spacing between targets to prevent wasting Novas on the same area.
        
        Args:
            position: Target position in game coordinates
            source_pending_id: ID of the pending target that is being confirmed (to avoid self-rejection)
            
        Returns:
            bool: True if registration successful, False if too close to existing target
        """
        try:
            # First check if this target is too close to an existing target
            for target_key, target_position in list(self.current_targets.items()):
                distance = position.distance_to(target_position)
                if distance < self.exclusion_radius * 2:
                    if self.debug_output:
                        print(f"DEBUG: Target at {position} rejected - too close to existing target at {target_position} (distance: {distance:.2f})")
                    return False
            
            # Also check against pending targets (except the one being confirmed)
            for target_id, target_position in self.pending_targets.items():
                # Skip checking against our own pending target
                if source_pending_id and target_id == source_pending_id:
                    if self.debug_output:
                        print(f"DEBUG: Skipping self-check for pending target {target_id}")
                    continue
                    
                distance = position.distance_to(target_position)
                if distance < self.exclusion_radius * 2:
                    if self.debug_output:
                        print(f"DEBUG: Target at {position} rejected - too close to pending target at {target_position} (distance: {distance:.2f})")
                    return False
                    
            # If we get here, the target is valid, so register it with a unique key
            target_key = f"target_{time.time()}_{position.x}_{position.y}"
            self.current_targets[target_key] = position
            if self.debug_output:
                print(f"DEBUG: Registered new Nova target at {position} with key {target_key}")
            return True
        except Exception as e:
            print(f"DEBUG ERROR registering Nova target: {e}")
            return False

    def unregister_nova_target(self, position: Point2) -> None:
        """
        Remove a target position from the registry when a Nova expires or is canceled.
        
        Handles approximate position matching to find the correct target to remove,
        as Nova positions may drift slightly during movement.
        
        Args:
            position: The approximate position to unregister
        """
        try:
            # Find closest match
            closest_key = None
            min_distance = float('inf')
            
            for target_key, target_position in list(self.current_targets.items()):
                distance = position.distance_to(target_position)
                
                # If within a reasonable threshold, consider it a match
                if distance < 15.0 and distance < min_distance:
                    closest_key = target_key
                    min_distance = distance
            
            # Remove the target if found
            if closest_key:
                actual_pos = self.current_targets[closest_key]
                del self.current_targets[closest_key]
                if self.debug_output:
                    print(f"DEBUG: Unregistered Nova target at {position}, closest match: {actual_pos}, distance: {min_distance:.2f}")
                return
                
            # If we get here without finding a target, look through all targets with a debug message
            if self.debug_output:
                if self.current_targets:
                    print(f"DEBUG: Could not find target near {position} to unregister. Current targets:")
                    for key, target in self.current_targets.items():
                        print(f"  Target at {target} (key: {key})")
                else:
                    print("DEBUG: No current Nova targets registered")
        except Exception as e:
            print(f"DEBUG ERROR unregistering Nova target: {e}")

    def get_exclusion_mask(self, grid) -> np.ndarray:
        """
        Generate a mask of areas that should be excluded from Nova targeting.
        
        Creates circular exclusion zones around all currently active Nova targets
        to prevent multiple Novas from being fired at the same location.
        
        Args:
            grid: Tactical ground grid defining the map dimensions
            
        Returns:
            np.ndarray: Boolean mask where True indicates excluded areas
        """
        try:
            # Very specific check - must be instance of np.ndarray
            if not isinstance(grid, np.ndarray):
                if self.debug_output:
                    print(f"DEBUG: get_exclusion_mask received invalid grid type: {type(grid)}")
                # Return an empty mask of default shape (100x100) if grid is invalid
                return np.zeros((100, 100), dtype=bool)
                
            # Get the shape of the grid for creating our mask
            grid_shape = grid.shape
            if self.debug_output:
                print(f"DEBUG: Creating exclusion mask with shape {grid_shape}")
                
            # Create an empty mask matching the grid shape
            mask = np.zeros(grid_shape, dtype=bool)
            
            # If we don't have any targets (current or pending), return the empty mask
            if not self.current_targets and not self.pending_targets:
                if self.debug_output:
                    print(f"DEBUG: No exclusion zones (no targets)")
                return mask
                
            # Mark exclusion zones around each target (both current and pending)
            all_targets = list(self.current_targets.items()) + list(self.pending_targets.items())
            for target_key, target_pos in all_targets:
                try:
                    # Extract x, y from the Point2 object
                    target_x, target_y = target_pos.x, target_pos.y
                        
                    # Create a circular mask at this position
                    mask_radius = self.exclusion_radius
                    
                    # Calculate grid coordinates for the circle
                    y_indices, x_indices = np.ogrid[:grid_shape[0], :grid_shape[1]]
                    # Calculate squared distance from this position
                    dist_from_target = ((x_indices - target_x)**2 + (y_indices - target_y)**2)
                    # Create circular mask
                    circular_mask = dist_from_target <= mask_radius**2
                    # Apply to our main mask
                    mask = np.logical_or(mask, circular_mask)
                    if self.debug_output:
                        print(f"DEBUG: Added exclusion zone at ({target_x}, {target_y}) with radius {mask_radius}")
                except Exception as e:
                    print(f"DEBUG ERROR creating exclusion zone for target {target_key}: {target_pos}, error: {e}")
        
            if self.debug_output:
                print(f"DEBUG: Generated exclusion mask with {np.sum(mask)} excluded cells")
            return mask
        except Exception as e:
            print(f"DEBUG ERROR in get_exclusion_mask: {e}")
            # Return empty mask on error
            return np.zeros((100, 100), dtype=bool)

    def can_nova_reach_target(self, nova_position: Point2, target_position: Point2, frames_left: int) -> bool:
        """
        Calculate if a Nova can reach a target before expiring.
        
        Uses Nova speed, remaining lifetime, and distance to determine
        if the target is within the Nova's maximum remaining travel range.
        
        Args:
            nova_position: Current Nova position
            target_position: Potential target position
            frames_left: Remaining lifetime in frames
            
        Returns:
            bool: True if target is reachable, False otherwise
        """
        #TODO change this into a sphere that shrinks as the nova expires
        # Calculate maximum travel distance - using the constant game steps per second (22.4) consistently
        max_travel_distance = self.nova_speed * (frames_left / 22.4)
        
        # Calculate current distance to target
        current_distance = nova_position.distance_to(target_position)
        
        # Return whether the target is reachable
        return current_distance <= max_travel_distance

    def update_units(self, enemy_units, friendly_units):
        """
        Update the cached unit lists for the current game step.
        
        Maintains references to current battlefield state for Nova targeting
        without requiring these lists to be passed to every method.
        
        Args:
            enemy_units: Current visible enemy units
            friendly_units: Current friendly units
        """
        self.enemy_units = enemy_units
        self.friendly_units = friendly_units

    def _clean_expired_target_entries(self, target_dict: Dict[str, Point2], max_age: float, target_type: str) -> None:
        """
        Helper method to clean expired targets from a target dictionary.
        
        Args:
            target_dict: Dictionary of targets to clean (current_targets or pending_targets)
            max_age: Maximum age in seconds before targets are considered expired
            target_type: Description of target type for debug messages (e.g., "active", "pending")
        """
        targets_to_remove = []
        current_time = time.time()
        
        # Find targets that have been active longer than the specified time
        for target_id, _ in target_dict.items():
            # Extract timestamp from the target ID if possible
            if '_' in target_id:
                try:
                    # Format is typically "type_timestamp_x_y"
                    parts = target_id.split('_')
                    if len(parts) >= 2:
                        timestamp = float(parts[1])
                        if current_time - timestamp > max_age:
                            targets_to_remove.append(target_id)
                except (ValueError, IndexError):
                    # If we can't parse the timestamp, leave it alone
                    pass
        
        # Remove the expired targets
        for target_id in targets_to_remove:
            del target_dict[target_id]
            if self.debug_output:
                print(f"DEBUG: Removed expired {target_type} target {target_id}")
    
    def clean_expired_targets(self) -> None:
        """
        Remove targets that have expired from the target registry.
        
        Handles cleanup of targets that are no longer active but weren't properly
        unregistered when their associated Nova expired.
        """
        try:
            # Clean active targets - allow 1.5x the nova lifetime for buffer
            self._clean_expired_target_entries(
                self.current_targets, 
                self.nova_lifetime * 1.5, 
                "active"
            )
            
            # Clean pending targets - these should be confirmed quickly (3 seconds)
            self._clean_expired_target_entries(
                self.pending_targets,
                3.0,
                "pending"
            )
                
        except Exception as e:
            print(f"DEBUG ERROR in clean_expired_targets: {e}")
            traceback.print_exc()