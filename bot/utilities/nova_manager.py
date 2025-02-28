from typing import List, TYPE_CHECKING

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

    def update(self, enemy_units: list, friendly_units: list) -> None:
        """Update all active nova instances and remove expired ones."""
        expired = []
        
        for nova in self.active_novas:
            # Decrement the frame counter and update remaining distance
            nova.update_info()
            
            # Run the nova's step logic
            nova.run_step(enemy_units, friendly_units)
            
            # If timer has expired, mark for removal
            if nova.frames_left <= 0:
                expired.append(nova)
        
        # Remove expired nova instances
        for nova in expired:
            self.active_novas.remove(nova)

    def get_active_novas(self) -> List:
        """Return the list of currently active nova instances."""
        return self.active_novas
