import numpy as np

from sc2.position import Point2
from sc2.units import Units
from sc2.unit import Unit
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId

from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    KeepUnitSafe, PathUnitToTarget, StutterUnitBack
)
from ares.behaviors.combat.group import (
    AMoveGroup, PathGroupToTarget
)
from ares.managers.squad_manager import UnitSquad
from ares.consts import UnitRole, UnitTreeQueryType

from cython_extensions import (
   
    cy_find_units_center_mass, cy_pick_enemy_target, cy_closest_to
    
)


# If you reference these from your bot.py, just import them directly from here
COMMON_UNIT_IGNORE_TYPES: set[UnitTypeId] = {
    UnitTypeId.EGG,
    UnitTypeId.LARVA,
    UnitTypeId.CREEPTUMORBURROWED,
    UnitTypeId.CREEPTUMORQUEEN,
    UnitTypeId.CREEPTUMOR,
    UnitTypeId.MULE,
    UnitTypeId.PROBE,
    UnitTypeId.SCV,
    UnitTypeId.DRONE,
    UnitTypeId.OVERLORD,
    UnitTypeId.OVERSEER,
    UnitTypeId.LOCUSTMP,
    UnitTypeId.LOCUSTMPFLYING,
    UnitTypeId.ADEPTPHASESHIFT,
    UnitTypeId.CHANGELING,
    UnitTypeId.CHANGELINGMARINE,
    UnitTypeId.CHANGELINGZEALOT,
    UnitTypeId.CHANGELINGZERGLING,
}


def control_main_army(bot, main_army: Units, target: Point2, squads: list[UnitSquad]) -> Point2:
    """
    Controls the main army's movement and engagement logic.
    """
    pos_of_main_squad: Point2 = bot.mediator.get_position_of_main_squad(role=UnitRole.ATTACKING)
    grid: np.ndarray = bot.mediator.get_ground_grid
    
    if main_army:
        bot.total_health_shield_percentage = (
            sum(unit.shield_health_percentage for unit in main_army) / len(main_army)
        )

    for squad in squads:
        maneuver = CombatManeuver()
        squad_position: Point2 = squad.squad_position
        units: list[Unit] = squad.squad_units
        squad_tags = squad.tags

        # Find nearby enemy units (excluding unimportant ones)
        all_close = bot.mediator.get_units_in_range(
            start_points=[squad_position],
            distances=13,
            query_tree=UnitTreeQueryType.AllEnemy,
            return_as_dict=False,
        )[0].filter(lambda u: not u.is_memory and not u.is_structure and u.type_id not in COMMON_UNIT_IGNORE_TYPES)

        # Basic engagement logic
        if all_close:
            # Separate melee and ranged
            melee = [u for u in units if u.ground_range <= 3]
            ranged = [u for u in units if u.ground_range > 3]
            # Simple picking logic
            enemy_target = cy_pick_enemy_target(all_close)

            # Ranged micro example
            for r_unit in ranged:
                ranged_maneuver = CombatManeuver()
                if r_unit.shield_health_percentage < 0.2:
                    ranged_maneuver.add(KeepUnitSafe(r_unit, grid))
                else:
                    ranged_maneuver.add(StutterUnitBack(r_unit, target=enemy_target, grid=grid))
                bot.register_behavior(ranged_maneuver)

            # Melee engages directly
            if melee:
                melee_maneuver = CombatManeuver()
                melee_maneuver.add(AMoveGroup(group=melee, group_tags={u.tag for u in melee}, target=enemy_target.position))
                bot.register_behavior(melee_maneuver)

        else:
            # No enemies nearby—regroup or move to final target
            dist_squad_main = pos_of_main_squad.distance_to(squad_position)
            if dist_squad_main > 0.1:
                maneuver.add(
                    PathGroupToTarget(
                        start=squad_position,
                        group=units,
                        group_tags=squad_tags,
                        target=pos_of_main_squad,
                        grid=grid,
                        sense_danger=False,
                        success_at_distance=0.1
                    )
                )
            else:
                maneuver.add(AMoveGroup(group=units, group_tags=squad_tags, target=target.position))

            bot.register_behavior(maneuver)

    return pos_of_main_squad


def warp_prism_follower(bot, warp_prisms: Units, main_army: Units) -> None:
    """
    Controls Warp Prisms: follows army, morphs between Transport/Phasing.
    """
    air_grid = bot.mediator.get_air_grid
    if not warp_prisms:
        return

    maneuver = CombatManeuver()
    for prism in warp_prisms:
        if main_army:
            distance_to_center = prism.distance_to(main_army.center)

            # If close, morph to Phasing
            if distance_to_center < 15 and prism.is_idle:
                prism(AbilityId.MORPH_WARPPRISMPHASINGMODE)
            else:
                # If no warpin in progress, revert to Transport
                # Or simply path near the army
                if prism.type_id == UnitTypeId.WARPPRISMPHASING:
                    not_ready_units = [unit for unit in bot.units if not unit.is_ready and unit.distance_to(prism) < 6.5]
                    if not not_ready_units:
                        prism(AbilityId.MORPH_WARPPRISMTRANSPORTMODE)

                # Keep prism ~5 distance behind the army center
                direction_vector = (prism.position - main_army.center).normalized
                new_target = main_army.center + direction_vector * 5
                maneuver.add(PathUnitToTarget(unit=prism, target=new_target, grid=air_grid, danger_distance=10))
        else:
            # If no main army, just retreat to natural (or wherever)
            maneuver.add(
                PathUnitToTarget(
                    unit=prism,
                    target=bot.natural_expansion,
                    grid=air_grid,
                    danger_distance=10
                )
            )

    bot.register_behavior(maneuver)


def assess_threat(bot, enemy_units: Units, own_forces: Units) -> int: #TODO check why units aren't responding to enemy
    """
    Assigns a 'threat level' based on unit composition.
    You can refine logic to suit your needs.
    """
    threat_level = 0
    for unit in enemy_units:
        if unit.type_id in (
            UnitTypeId.MARINE, UnitTypeId.ZEALOT, UnitTypeId.ZERGLING,
            UnitTypeId.ADEPT, UnitTypeId.STALKER, UnitTypeId.ROACH,
            UnitTypeId.REAPER, UnitTypeId.MARAUDER, UnitTypeId.SENTRY,
            UnitTypeId.HYDRALISK, UnitTypeId.BANELING, UnitTypeId.HELLION,
            UnitTypeId.HELLIONTANK, UnitTypeId.HIGHTEMPLAR, UnitTypeId.MUTALISK,
            UnitTypeId.BANSHEE, UnitTypeId.VIKING, UnitTypeId.VIKINGFIGHTER,
            UnitTypeId.PHOENIX, UnitTypeId.ORACLE, UnitTypeId.RAVEN, UnitTypeId.GHOST
        ):
            threat_level += 2
        elif unit.type_id in (
            UnitTypeId.SIEGETANK, UnitTypeId.IMMORTAL, UnitTypeId.CYCLONE,
            UnitTypeId.DISRUPTOR, UnitTypeId.COLOSSUS, UnitTypeId.RAVAGER,
            UnitTypeId.LURKER, UnitTypeId.VOIDRAY, UnitTypeId.CARRIER,
            UnitTypeId.BATTLECRUISER, UnitTypeId.TEMPEST, UnitTypeId.BROODLORD,
            UnitTypeId.ULTRALISK, UnitTypeId.THOR, UnitTypeId.SIEGETANKSIEGED,
            UnitTypeId.LIBERATOR, UnitTypeId.LIBERATORAG, UnitTypeId.LURKERBURROWED,
            UnitTypeId.DARKTEMPLAR, UnitTypeId.ARCHON, UnitTypeId.CORRUPTOR,
            UnitTypeId.WIDOWMINE, UnitTypeId.INFESTOR, UnitTypeId.INFESTORBURROWED,
            UnitTypeId.SWARMHOSTBURROWEDMP, UnitTypeId.VIPER, UnitTypeId.WIDOWMINEBURROWED
        ):
            threat_level += 3
        else:
            threat_level += 1

    # Simple “we have bigger army” adjustment
    if own_forces.amount > enemy_units.amount * 2:
        threat_level -= 2

    return max(threat_level, 0)


def threat_detection(bot, main_army: Units) -> None:
    """
    Detects if we have ground or air enemies near bases; calls 
    assess_threat and possibly redirects the main army to defend.
    """
    ground_near = bot.mediator.get_ground_enemy_near_bases
    flying_near = bot.mediator.get_flying_enemy_near_bases

    if ground_near or flying_near:
        # Combine ground + air threats
        all_threats = {}
        for key, value in ground_near.items():
            all_threats[key] = value.copy()
        for key, value in flying_near.items():
            all_threats.setdefault(key, set()).update(value)

        for _, enemy_tags in all_threats.items():
            enemy_units = bot.enemy_units.tags_in(enemy_tags)
            threat_value = assess_threat(bot, enemy_units, main_army)

            # Early game threat detection
            if bot.time < 2 * 60 + 20 and bot.townhalls.first:
                unit_categories = {'pylons': [], 'enemyWorkerUnits': [], 'cannons': [], 'zerglings': []}
                for _, enemy_tags in ground_near.items():
                    enemy_units = bot.enemy_units.tags_in(enemy_tags)
                    for unit in enemy_units:
                        if unit.type_id == UnitTypeId.PYLON:
                            unit_categories['pylons'].append(unit)
                            print("Pylon Detected")
                        elif unit.type_id in [UnitTypeId.PROBE, UnitTypeId.SCV, UnitTypeId.DRONE]:
                            unit_categories['enemyWorkerUnits'].append(unit)
                        elif unit.type_id == UnitTypeId.PHOTONCANNON:
                            unit_categories['cannons'].append(unit)
                            print("Cannon Detected")
                        elif unit.type_id == UnitTypeId.ZERGLING:
                            unit_categories['zerglings'].append(unit)
                            print("Zergling Detected")

                if unit_categories['pylons'] or len(unit_categories['enemyWorkerUnits']) >= 4 or unit_categories['cannons']:
                    bot.defend_worker_cannon_rush(unit_categories['enemyWorkerUnits'], unit_categories['cannons'])
                    print("cannon rush")
                elif len(unit_categories['zerglings']) > 2:
                    bot._used_cheese_response = True
                    print("Defending against zergling rush")

            # If there's a threat and we have a main army, send the army to defend
            if main_army:
                threat_position, num_units = cy_find_units_center_mass(enemy_units, 10.0)
                threat_position = Point2(threat_position)
                if bot.time < 3 * 60 and bot._used_cheese_response:
                    if assess_threat(bot, enemy_units, main_army) >= 2:
                        bot._under_attack = True
                elif bot._under_attack and assess_threat(bot, enemy_units, main_army) < 2:
                    bot._under_attack = False
                elif not bot._under_attack and assess_threat(bot, enemy_units, main_army) >= 5:
                    bot._under_attack = True
                else:
                    bot._under_attack = False
                if bot._under_attack:
                    control_main_army(bot, main_army, threat_position, bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=15))
                    if bot._one_base_reaction_completed:
                        bot.use_overcharge(main_army)


def handle_attack_toggles(bot, main_army: Units, attack_target: Point2) -> None:
    """
    Handles the attack toggles logic.
    """
    current_supply = bot.get_total_supply(main_army)
    if current_supply <= bot._begin_attack_at_supply:
        bot._commenced_attack = False
    elif bot._commenced_attack and not bot._under_attack:
        control_main_army(bot, main_army, attack_target, bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=15))
    elif current_supply >= bot._begin_attack_at_supply:
        bot._commenced_attack = True


def attack_target(bot, main_army_position: Point2) -> Point2:
    """
    Determines where our main attacking units should move toward.
    If we see an enemy structure, we target the closest one. Otherwise,
    we may fallback to expansions or the enemy start location.
    """

    if bot.enemy_structures:
        # Prioritize the closest enemy structure to the main army
        closest_structure = cy_closest_to(main_army_position, bot.enemy_structures).position
        
        # Check if the closest structure is far away and if so, fallback to a stable target
        if closest_structure.distance_to(main_army_position) > 25.0:
            return fallback_target(bot)
        
        return Point2(closest_structure.position)
        
    # Not seen anything in early game, just head to enemy spawn
    elif bot.time < 240.0:
        return bot.enemy_start_locations[0]
    
    # Else search the map
    else:
        return fallback_target(bot)


def fallback_target(bot) -> Point2:
    """
    Provides a fallback target for the main army when no clear target is available.
    Cycles through expansion locations.
    """
    # Cycle through expansion locations
    if bot.is_visible(bot.current_base_target):
        bot.current_base_target = next(bot.expansions_generator)
    
    return bot.current_base_target
