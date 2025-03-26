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

    def __init__(self, bot, mediator):
        """Initialize the Nova Manager.
        
        Args:
            bot: Reference to the main bot instance
            mediator: Access point for game state and tactical information
        """
        # Core references
        self.bot = bot
        self.mediator = mediator
        
        # Nova tracking
        self.active_novas: List = []
        self.current_targets: Dict[str, Point2] = {}
        
        # Nova constants
        self.nova_speed = 4.25  # Movement speed (game units/second)
        self.nova_lifetime = 2.1  # Lifetime (seconds)
        self.nova_radius = 1.5  # Explosion radius
        
        # Targeting parameters
        self.exclusion_radius = 3.0  # Grid cells
        self.exclusion_radius_game_units = 3.0  # Game units
        
        # Unit tracking
        self.enemy_units = []
        self.friendly_units = []
        self.current_time = time.time()

        print(f"DEBUG: NovaManager initialized with exclusion radius of {self.exclusion_radius} grid cells")

        try:
            grid = self.mediator.get_tactical_ground_grid
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
            
            # Track and update active nova instances
            expired_novas = []
            
            for nova in self.active_novas:
                # Decrement the frame counter and update remaining distance
                nova.update_info()
                
                # Run the nova's step logic with target adjustment
                nova.run_step(self.enemy_units, self.friendly_units, self)
                
                # If timer has expired, mark for removal
                if nova.frames_left <= 0:
                    expired_novas.append(nova)
                    
                    # When a nova expires, unregister its target
                    if hasattr(nova, 'best_target_pos') and nova.best_target_pos:
                        self.unregister_nova_target(nova.best_target_pos)
                        print(f"DEBUG: Nova at {nova.best_target_pos} expired, target unregistered")
            
            # Remove expired novas
            if expired_novas:
                for nova in expired_novas:
                    self.active_novas.remove(nova)
                print(f"DEBUG: Removed {len(expired_novas)} expired novas, {len(self.active_novas)} still active")
                
            # Debug info about current tracking state
            if self.active_novas:
                print(f"DEBUG: Currently tracking {len(self.active_novas)} active novas and {len(self.current_targets)} registered targets")
                
        except Exception as e:
            print(f"DEBUG ERROR in NovaManager.update: {e}")
            traceback.print_exc()

    def get_active_novas(self) -> List:
        """Get all currently active Nova instances.
        
        Returns:
            List: Active UseDisruptorNova behavior instances
        """
        return self.active_novas
        
    def register_nova_target(self, position: Point2) -> bool:
        """
        Register a position as a Nova target to prevent overlapping Novas.
        
        Maintains a registry of active Nova targets and enforces minimum
        spacing between targets to prevent wasting Novas on the same area.
        
        Args:
            position: Target position in game coordinates
            
        Returns:
            bool: True if registration successful, False if too close to existing target
        """
        try:
            # Validate input
            if not position:
                print("DEBUG: Cannot register None position")
                return False
            
            # Check if this is effectively the same as an existing target
            for target_key, target_position in list(self.current_targets.items()):
                distance = position.distance_to(target_position)
                # If very close (almost the same point), consider it already registered
                if distance < 0.5:
                    # No need to log here - reduces spam
                    return True
                    
                # If within exclusion radius but not identical, reject as too close
                if distance < self.exclusion_radius_game_units:
                    print(f"DEBUG: Nova target registration failed - target at {position} too close to existing target at {target_position}")
                    print(f"DEBUG: Distance: {distance:.2f}, Minimum required: {self.exclusion_radius_game_units:.2f}")
                    return False
            
            # Generate a unique key for this target
            target_key = f"nova_{len(self.current_targets)}_{int(self.current_time)}"
            
            # Register the target
            self.current_targets[target_key] = position
            print(f"DEBUG: Registered Nova target at {position}, now tracking {len(self.current_targets)} targets")
            return True
            
        except Exception as e:
            print(f"DEBUG ERROR in register_nova_target: {e}")
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
                print(f"DEBUG: Unregistered Nova target at {position}, closest match: {actual_pos}, distance: {min_distance:.2f}")
                return
                
            # If we get here without finding a target, look through all targets with a debug message
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
                print(f"DEBUG: get_exclusion_mask received invalid grid type: {type(grid)}")
                # Return an empty mask of default shape (100x100) if grid is invalid
                return np.zeros((100, 100), dtype=bool)
                
            # Get the shape of the grid for creating our mask
            grid_shape = grid.shape
            print(f"DEBUG: Creating exclusion mask with shape {grid_shape}")
                
            # Create an empty mask matching the grid shape
            mask = np.zeros(grid_shape, dtype=bool)
            
            # If we don't have any current targets, return the empty mask
            if not self.current_targets:
                print(f"DEBUG: No exclusion zones (no current targets)")
                return mask
                
            # Mark exclusion zones around each current target
            for target_key, target_pos in self.current_targets.items():
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
                    print(f"DEBUG: Added exclusion zone at ({target_x}, {target_y}) with radius {mask_radius}")
                except Exception as e:
                    print(f"DEBUG ERROR creating exclusion zone for target {target_key}: {target_pos}, error: {e}")
        
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
        # Calculate maximum travel distance - using the constant FPS value consistently
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