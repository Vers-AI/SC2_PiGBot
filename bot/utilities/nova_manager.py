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
        # Set to track current Nova targets and avoid overlapping
        self.current_targets: Set[Tuple[int, int]] = set()
        # Exclusion radius (in grid cells)
        self.exclusion_radius = 3  # Reduced from 5 to 3 to be less restrictive
        # Nova speed and lifetime constants
        self.nova_speed = 5.95  # Nova movement speed in game units per second
        self.nova_lifetime = 2.1  # Nova lifetime in seconds
        self.enemy_units = []
        self.friendly_units = []

        print("DEBUG: NovaManager initialized with exclusion radius of 3 grid cells")

        # Test to verify we can access the tactical grid
        try:
            # Get the tactical ground grid as a method
            grid = self.mediator.get_tactical_ground_grid()
            if grid is not None:
                print(f"DEBUG: Successfully accessed tactical grid with shape {grid.shape}")
            else:
                print("DEBUG: Warning - tactical grid is None")
        except Exception as e:
            print(f"DEBUG ERROR initializing NovaManager - could not access tactical grid: {e}")

    def add_nova(self, nova) -> None:
        """Add a nova instance to the manager."""
        # Import here to avoid circular import
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
                # Convert target position to grid coordinates
                grid_x, grid_y = int(nova.best_target_pos.x), int(nova.best_target_pos.y)
                # Remove from tracked targets
                if (grid_x, grid_y) in self.current_targets:
                    self.current_targets.remove((grid_x, grid_y))
            self.active_novas.remove(nova)

    def get_active_novas(self) -> List:
        """Return the list of currently active nova instances."""
        return self.active_novas
        
    def register_nova_target(self, position: Point2) -> bool:
        """
        Register a new Nova target position, checking for overlap with existing targets.
        
        Args:
            position: The target position (Point2)
            
        Returns:
            bool: True if successfully registered, False if too close to existing target
        """
        try:
            # Validate input
            if position is None:
                print("DEBUG: register_nova_target called with None position")
                return False
                
            # Convert Point2 to grid coordinates
            # For simplicity, assuming grid coordinates match game coordinates
            grid_x = int(position.x)
            grid_y = int(position.y)
            
            # Check if too close to existing target
            for existing_x, existing_y in self.current_targets:
                distance_squared = (existing_x - grid_x) ** 2 + (existing_y - grid_y) ** 2
                exclusion_distance_squared = (self.exclusion_radius * 2) ** 2
                
                if distance_squared <= exclusion_distance_squared:
                    actual_distance = np.sqrt(distance_squared)
                    exclusion_distance = self.exclusion_radius * 2
                    print(f"DEBUG: Nova target registration failed - target at ({grid_x}, {grid_y}) too close to existing target at ({existing_x}, {existing_y})")
                    print(f"DEBUG: Distance: {actual_distance:.2f}, Minimum required: {exclusion_distance:.2f}")
                    return False
                
            # Add to current targets
            self.current_targets.add((grid_x, grid_y))
            print(f"DEBUG: Registered Nova target at ({grid_x}, {grid_y}), now tracking {len(self.current_targets)} targets")
            return True
        except Exception as e:
            print(f"DEBUG ERROR in register_nova_target: {e}")
            return False
            
    def unregister_nova_target(self, position: Point2) -> None:
        """
        Remove a Nova target position from tracking.
        
        Args:
            position: The position of the Nova target in the game world
        """
        try:
            # Validate input
            if position is None:
                print("DEBUG: unregister_nova_target called with None position")
                return
                
            # Convert Point2 to grid coordinates
            # For simplicity, assuming grid coordinates match game coordinates
            grid_x = int(position.x)
            grid_y = int(position.y)
            
            # Remove from current targets
            if (grid_x, grid_y) in self.current_targets:
                self.current_targets.remove((grid_x, grid_y))
                print(f"DEBUG: Unregistered Nova target at ({grid_x}, {grid_y}), now tracking {len(self.current_targets)} targets")
            else:
                print(f"DEBUG: Nova target at ({grid_x}, {grid_y}) not found in registry")
        except Exception as e:
            print(f"DEBUG ERROR in unregister_nova_target: {e}")

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
            print(f"DEBUG: Creating exclusion mask with shape {grid_shape}")
                
            # Create an empty mask matching the grid shape
            mask = np.zeros(grid_shape, dtype=bool)
            
            # If we don't have any current targets, return the empty mask
            if not self.current_targets:
                print(f"DEBUG: No exclusion zones (no current targets)")
                return mask
                
            # Mark exclusion zones around each current target
            for target_pos in self.current_targets:
                try:
                    # current_targets should now contain (x, y) tuples
                    target_x, target_y = target_pos
                        
                    # Create a circular mask at this position
                    mask_radius = self.exclusion_radius
                    
                    # Safety check for grid boundaries
                    if (0 <= target_y < grid_shape[0] and 0 <= target_x < grid_shape[1]):
                        # Calculate grid coordinates for the circle
                        y_indices, x_indices = np.ogrid[:grid_shape[0], :grid_shape[1]]
                        # Calculate squared distance from this position
                        dist_from_target = ((x_indices - target_x)**2 + (y_indices - target_y)**2)
                        # Create circular mask
                        circular_mask = dist_from_target <= mask_radius**2
                        # Apply to our main mask
                        mask = np.logical_or(mask, circular_mask)
                        print(f"DEBUG: Added exclusion zone at ({target_x}, {target_y}) with radius {mask_radius}")
                    else:
                        print(f"DEBUG: Target position ({target_x}, {target_y}) out of grid bounds {grid_shape}")
                except Exception as e:
                    print(f"DEBUG ERROR creating exclusion zone for target {target_pos}: {e}")
                
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
        # Convert frames to seconds
        seconds_left = frames_left / 22.4  # Assuming 22.4 frames per second
        
        # Calculate maximum travel distance
        max_travel_distance = self.nova_speed * seconds_left
        
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
