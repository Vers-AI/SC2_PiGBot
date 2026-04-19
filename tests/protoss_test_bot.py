"""Protoss Test Bot — Modular Protoss opponent for testing threat detection.

Purpose: Controlled Protoss opponent with selectable behavior modules.
         Each module tests a specific threat type that PiGBot needs to handle.
Key Decisions: Minimal economy, behavior modules are toggled via class attributes.
Limitations: No defense, no upgrades beyond what modules require.

Usage: --TestBot=Protoss (default: all modules active)

Modules:
  - Worker Rush: Send all workers to attack enemy start location
  - (future: DT rush, Oracle harass, etc.)
"""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId


class ProtossTestBot(BotAI):
    """Protoss test bot with modular threat behaviors.

    Set class attributes to enable/disable specific test modules:
      - enable_worker_rush: Send all Probes to attack (default True)
    """

    enable_worker_rush: bool = True

    async def on_step(self, iteration: int):
        if iteration == 0:
            for worker in self.workers:
                worker.gather(self.mineral_field.closest_to(worker))

        # --- Behavior modules ---
        if self.enable_worker_rush:
            self._run_worker_rush()

    def _run_worker_rush(self) -> None:
        """Worker Rush module: A-move all Probes to enemy start location."""
        workers = self.units(UnitTypeId.PROBE).idle
        if not workers:
            workers = self.units(UnitTypeId.PROBE)
        if not workers:
            return

        if not self.enemy_start_locations:
            return

        enemy_start = self.enemy_start_locations[0]
        for worker in workers:
            worker.attack(enemy_start)