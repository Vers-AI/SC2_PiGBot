from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from bot.utilities.use_disruptor_nova import UseDisruptorNova


class NovaManager:
    """Manager for tracking and updating active Disruptor Nova abilities."""

    def __init__(self, bot, map_data):
        # Store bot and map_data for wrapping incoming nova units
        self.bot = bot
        self.map_data = map_data
        # List to hold active nova instances
        self.active_novas: List = []

    def add_nova(self, nova) -> None:
        """Add a nova instance to the manager."""
        # Import here to avoid circular import
        from bot.utilities.use_disruptor_nova import UseDisruptorNova
        
        if not hasattr(nova, 'update_info'):
            nova_instance = UseDisruptorNova(self.map_data, self.bot)
            nova_instance.load_info(nova)
            self.active_novas.append(nova_instance)
        else:
            self.active_novas.append(nova)

    def update(self, enemy_units: list, friendly_units: list) -> None:
        """Update all active nova instances and remove expired ones."""
        expired = []
        for nova in self.active_novas:
            # Update nova state
            nova.update_info()
            # Run the nova behavior step (which updates target and moves nova)
            nova.run_step(enemy_units, friendly_units)
            # Check if nova has expired
            if nova.frames_left <= 0:
                expired.append(nova)

        # Remove expired novas
        for nova in expired:
            self.active_novas.remove(nova)

    def get_active_novas(self) -> List:
        """Return the list of currently active nova instances."""
        return self.active_novas
