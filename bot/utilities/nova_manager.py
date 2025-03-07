from typing import List, Dict, TYPE_CHECKING, Optional, Set, Tuple
from sc2.ids.unit_typeid import UnitTypeId
import time
import numpy as np
from sc2.position import Point2

if TYPE_CHECKING:
    from bot.utilities.use_disruptor_nova import UseDisruptorNova


class NovaManager:
    """Manager for tracking and updating active Disruptor Nova abilities."""

    def __init__(self, bot, mediator):
        # Store bot and mediator for wrapping incoming nova units
        self.bot = bot
        self.mediator = mediator
        # List to hold active nova instances
        self.active_novas: List = []
        # Store full target positions instead of just coordinates for better matching
        self.current_targets: Dict[Tuple[float, float], Point2] = {}
        # Exclusion radius (in grid cells)
        self.exclusion_radius = 2.5  # Reduced from 3 to 2.5 for even more flexibility
        # Nova speed and lifetime constants
        self.nova_speed = 5.95  # Nova movement speed in game units per second
        self.nova_lifetime = 2.1  # Nova lifetime in seconds
        self.nova_radius = 1.5  # Nova radius
        self.enemy_units = []
        self.friendly_units = []

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
        """Add a nova instance to the manager."""
        from bot.utilities.use_disruptor_nova import UseDisruptorNova
        
        if not hasattr(nova, 'update_info'):
            nova_instance = UseDisruptorNova(mediator=self.mediator, bot=self.bot)
            nova_instance.load_info(nova)
            self.active_novas.append(nova_instance)
        else:
            self.active_novas.append(nova)
            
        # Register the target position for this nova
        if hasattr(nova, 'best_target_pos') and nova.best_target_pos:
            self.register_nova_target(nova.best_target_pos)

    
    def update(self, enemy_units: list, friendly_units: list) -> None:
        """Update all active nova instances and remove expired ones."""
        self.update_units(enemy_units, friendly_units)
        expired = []
        
        for nova in self.active_novas:
            # Decrement the frame counter and update remaining distance
            nova.update_info()
            
            # Run the nova's step logic with target adjustment
            nova.run_step(self.enemy_units, self.friendly_units, self)
            
            # If timer has expired, mark for removal
            if nova.frames_left <= 0:
                expired.append(nova)
        
        # Remove expired nova instances
        for nova in expired:
            if hasattr(nova, 'best_target_pos') and nova.best_target_pos:
                # Use the proper unregister method to ensure consistency
                try:
                    self.unregister_nova_target(nova.best_target_pos)
                except Exception as e:
                    print(f"DEBUG ERROR unregistering expired Nova target: {e}")
            self.active_novas.remove(nova)

                
    def get_active_novas(self) -> List:
        """Return the list of currently active nova instances."""
        return self.active_novas
        
    def register_nova_target(self, position: Point2) -> bool:
        """
        Register a position as a current Nova target.
        
        Args:
            position: The position (x, y) to register as target
            
        Returns:
            bool: True if registration was successful, False if target is too close to existing targets
        """
        try:
            # Check if this is effectively the same as an existing target
            for target_key, target_position in list(self.current_targets.items()):
                distance = position.distance_to(target_position)
                # If very close (almost the same point), consider it already registered
                if distance < 0.5:
                    # No need to log here - reduces spam
                    return True
                    
                # If within exclusion radius but not identical, reject as too close
                if distance < self.nova_radius:
                    print(f"DEBUG: Nova target at {position} is too close to existing target")
                    return False
            
            # Add position to dictionary of current targets
            # Use a tuple of rounded coordinates as the key for consistent lookup
            rounded_key = (round(position.x, 2), round(position.y, 2))
            self.current_targets[rounded_key] = position
            print(f"DEBUG: Registered Nova target at {position}, key: {rounded_key}")
            return True
        except Exception as e:
            print(f"DEBUG ERROR registering Nova target: {e}")
            return False

    def unregister_nova_target(self, position: Point2) -> None:
        """
        Remove a Nova target position from tracking.
        
        Args:
            position: The position to remove
        """
        try:
            # Use rounded position for lookup to handle small differences
            rounded_pos = (round(position.x, 2), round(position.y, 2))
            
            # Try direct lookup with rounded position first
            if rounded_pos in self.current_targets:
                # Get the original Point2 to report in logging
                actual_pos = self.current_targets[rounded_pos]
                del self.current_targets[rounded_pos]
                print(f"DEBUG: Directly unregistered Nova target at {position}, match: {actual_pos}")
                return
                
            # If no direct match, find closest match
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
                    print(f"  Target {key}: {target}, distance: {position.distance_to(target):.2f}")
            
        except Exception as e:
            print(f"DEBUG ERROR unregistering Nova target: {e}")

    def get_exclusion_mask(self, grid) -> np.ndarray:
        """
        Create a boolean mask of areas that should be excluded from targeting.
        
        Args:
            grid: The tactical ground grid
            
        Returns:
            np.ndarray: Boolean mask where True indicates areas to avoid
        """
        try:
            # Very specific check - must be instance of np.ndarray
            if not isinstance(grid, np.ndarray):
                print(f"DEBUG: get_exclusion_mask received invalid grid type: {type(grid)}")
                # Return an empty mask of default shape (100x100) if grid is invalid
                return np.zeros((100, 100), dtype=bool)
                
            # Get the shape of the grid for creating our mask
            grid_shape = grid.shape
            
            # Create an empty mask matching the grid shape
            mask = np.zeros(grid_shape, dtype=bool)
            
            # If we don't have any current targets, return the empty mask
            if not self.current_targets:
                return mask
                
            # More efficient approach: create a single mask in one pass
            # Mark exclusion zones around each current target
            if self.current_targets:
                # Create coordinate grids once for more efficient distance calculations
                x_indices, y_indices = np.indices(grid_shape)
                
                # For each target, create a circular mask and combine them all
                for _, target_position in self.current_targets.items():
                    target_x, target_y = int(target_position.x), int(target_position.y)
                    
                    # Safety check for grid boundaries
                    if (0 <= target_x < grid_shape[1] and 0 <= target_y < grid_shape[0]):
                        # Calculate squared distance from this position
                        dist_from_target = ((x_indices - target_x)**2 + (y_indices - target_y)**2)
                        # Create circular mask and combine with our existing mask
                        mask = np.logical_or(mask, dist_from_target <= self.exclusion_radius**2)
                    else:
                        print(f"DEBUG: Target position ({target_x}, {target_y}) out of grid bounds {grid_shape}")
                 
            print(f"DEBUG: Generated exclusion mask with {np.sum(mask)} excluded cells")
            return mask
        except Exception as e:
            print(f"DEBUG ERROR in get_exclusion_mask: {e}")
            # Return empty mask on error
            return np.zeros((100, 100), dtype=bool)

    def can_nova_reach_target(self, nova_position: Point2, target_position: Point2, frames_left: int) -> bool:
        """
        Check if a Nova can reach a target position within its remaining lifetime.
        
        Args:
            nova_position: Current position of the Nova
            target_position: Target position to check
            frames_left: Number of frames left in the Nova's lifetime
            
        Returns:
            bool: True if the Nova can reach the target, False otherwise
        """
        # Calculate maximum travel distance - using the constant FPS value consistently
        max_travel_distance = self.nova_speed * (frames_left / 22.4)
        
        # Calculate current distance to target
        current_distance = nova_position.distance_to(target_position)
        
        # Return whether the target is reachable
        return current_distance <= max_travel_distance

    def update_units(self, enemy_units, friendly_units):
        """
        Update the internal record of enemy and friendly units for this frame.
        
        Args:
            enemy_units: List of enemy units
            friendly_units: List of friendly units
        """
        self.enemy_units = enemy_units
        self.friendly_units = friendly_units