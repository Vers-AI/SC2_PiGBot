"""Zerg Test Bot — Modular Zerg opponent for testing threat detection.

Purpose: Controlled Zerg opponent with selectable behavior modules.
         Each module tests a specific threat type that PiGBot needs to handle.
Key Decisions: Minimal economy, behavior modules are toggled via class attributes.
Limitations: No defense, no upgrades beyond what modules require.

Usage: --TestBot=Zerg (default: all modules active)

Modules:
  - Fungal Growth: Infestors that cast Fungal on enemy clumps
  - (future: Baneling, Brood Lord, etc.)
"""

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId


class ZergTestBot(BotAI):
    """Zerg test bot with modular threat behaviors.

    Set class attributes to enable/disable specific test modules:
      - enable_fungal: Build Infestors and cast Fungal Growth (default True)
    """

    enable_fungal: bool = True

    async def on_step(self, iteration: int):
        if iteration == 0:
            for worker in self.workers:
                worker.gather(self.mineral_field.closest_to(worker))

        cc = self.townhalls.first
        if cc is None:
            return

        # --- Shared economy ---
        await self._run_economy(cc)

        # --- Behavior modules ---
        if self.enable_fungal:
            await self._run_fungal(cc)

    async def _run_economy(self, cc) -> None:
        """Minimal Zerg economy: Drones, Overlords, Extractors."""
        if cc.is_idle and self.can_afford(UnitTypeId.DRONE) and self.supply_left > 0:
            cc.train(UnitTypeId.DRONE)

        if self.supply_left < 4 and not self.already_pending(UnitTypeId.OVERLORD):
            if self.can_afford(UnitTypeId.OVERLORD):
                cc.train(UnitTypeId.OVERLORD)

        if self.structures.of_type(UnitTypeId.EXTRACTOR).amount < 2:
            if self.can_afford(UnitTypeId.EXTRACTOR):
                vgs = self.vespene_geyser.closer_than(15, cc)
                if vgs:
                    for vg in vgs:
                        if not self.structures.of_type(UnitTypeId.EXTRACTOR).closer_than(1, vg):
                            await self.build(UnitTypeId.EXTRACTOR, near=vg)
                            break

        for drone in self.workers.idle:
            if self.structures.of_type(UnitTypeId.EXTRACTOR).ready:
                ref = self.structures.of_type(UnitTypeId.EXTRACTOR).ready.first
                if ref.surplus_harvesters < 0:
                    drone.gather(ref)
                    continue
            drone.gather(self.mineral_field.closest_to(drone))

    async def _run_fungal(self, cc) -> None:
        """Fungal Growth module: build Infestors and cast on enemy clumps."""
        # Build Spawning Pool (prerequisite for Lair)
        if not self.structures.of_type(UnitTypeId.SPAWNINGPOOL) and self.can_afford(UnitTypeId.SPAWNINGPOOL):
            await self.build(UnitTypeId.SPAWNINGPOOL, near=cc.position.towards(self.game_info.map_center, 8))

        # Upgrade to Lair (prerequisite for Infestation Pit)
        if (
            self.structures.of_type(UnitTypeId.SPAWNINGPOOL).ready
            and not self.structures.of_type(UnitTypeId.LAIR)
            and not self.already_pending(UnitTypeId.LAIR)
            and self.can_afford(UnitTypeId.LAIR)
        ):
            cc.build(UnitTypeId.LAIR)

        # Build Infestation Pit (prerequisite for Infestor)
        if (
            self.structures.of_type(UnitTypeId.LAIR).ready
            and not self.structures.of_type(UnitTypeId.INFESTATIONPIT)
            and self.can_afford(UnitTypeId.INFESTATIONPIT)
        ):
            await self.build(UnitTypeId.INFESTATIONPIT, near=cc.position.towards(self.game_info.map_center, 10))

        # Produce Infestors from Larva
        if self.structures.of_type(UnitTypeId.INFESTATIONPIT).ready:
            if self.can_afford(UnitTypeId.INFESTOR) and self.supply_left > 0:
                larvae = self.larva
                if larvae:
                    larvae.random.train(UnitTypeId.INFESTOR)

        # Infestor micro: move toward enemy and cast Fungal
        infestors = self.units(UnitTypeId.INFESTOR)
        if not infestors:
            return

        enemy_units = self.enemy_units
        if not enemy_units:
            if self.enemy_start_locations:
                target = self.enemy_start_locations[0]
                for inf in infestors:
                    inf.move(target)
            return

        for inf in infestors:
            if inf.energy >= 75:
                # Find best fungal target: largest clump of enemies
                best_target = None
                best_count = 0
                for enemy in enemy_units:
                    count = len(enemy_units.closer_than(2.0, enemy))
                    if count > best_count:
                        best_count = count
                        best_target = enemy.position

                if best_target is not None and best_count >= 2:
                    inf(AbilityId.FUNGALGROWTH_FUNGALGROWTH, best_target)
                else:
                    closest_enemy = enemy_units.closest_to(inf)
                    inf.move(closest_enemy.position.towards(inf.position, -5))
            else:
                # Not enough energy — move toward enemy
                closest_enemy = enemy_units.closest_to(inf)
                dist = inf.distance_to(closest_enemy)
                if dist > 10:
                    inf.move(closest_enemy.position.towards(inf.position, -5))