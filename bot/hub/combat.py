import numpy as np

from sc2.position import Point2
from sc2.units import Units
from sc2.unit import Unit
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.ability_id import AbilityId
from ares.consts import UnitTreeQueryType

from ares.behaviors.combat import CombatManeuver
from ares.behaviors.combat.individual import (
    KeepUnitSafe, PathUnitToTarget, StutterUnitBack, UseAbility
)
from ares.behaviors.combat.group import (
    AMoveGroup, PathGroupToTarget
)
from ares.managers.squad_manager import UnitSquad
from ares.consts import UnitRole
from ares.dicts.unit_data import UNIT_DATA

from bot.utilities.use_disruptor_nova import UseDisruptorNova

from cython_extensions import (
   
    cy_find_units_center_mass, cy_pick_enemy_target, cy_closest_to
    
)


#Units to ignore
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
            # Separate melee, ranged and spell casters
            melee = [u for u in units if u.ground_range <= 3]
            ranged = [u for u in units if u.ground_range > 3 and u.energy == 0]
            spellcasters = [u for u in units if u.energy > 0 or not u.can_attack]
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
            
            if spellcasters:
                disruptors= [spellcaster for spellcaster in spellcasters if spellcaster.type_id == UnitTypeId.DISRUPTOR]
                for disruptor in disruptors:
                        bot.use_disruptor_nova.execute(disruptor, all_close, units)
                        bot.register_behavior(KeepUnitSafe(disruptor, grid))
                
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
                maneuver.add(AMoveGroup(group=units, group_tags=squad_tags, target=target))

            bot.register_behavior(maneuver)

    return pos_of_main_squad


def warp_prism_follower(bot, warp_prisms: Units, main_army: Units) -> None:
    """
    Controls Warp Prisms: follows army, morphs between Transport/Phasing.
    """
    air_grid: np.ndarray = bot.mediator.get_air_grid
    if not warp_prisms:
        return

    maneuver: CombatManeuver = CombatManeuver()
    for prism in warp_prisms:
        if main_army:
            distance_to_center = prism.distance_to(main_army.center)

            # If close, morph to Phasing
            if distance_to_center < 15:
                prism(AbilityId.MORPH_WARPPRISMPHASINGMODE)
            else:
                # If no warpin in progress, revert to Transport
                # Or simply path near the army
                not_ready_units = [unit for unit in bot.units if not unit.is_ready and unit.distance_to(prism) < 6.5]
                if prism.type_id == UnitTypeId.WARPPRISMPHASING and not not_ready_units:
                        prism(AbilityId.MORPH_WARPPRISMTRANSPORTMODE)

                elif prism.type_id == UnitTypeId.WARPPRISM:
                    # Keep prism ~3 distance behind the army center
                    direction_vector = (prism.position - main_army.center).normalized
                    new_target = main_army.center + direction_vector * 3
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


def assess_threat(bot, enemy_units: Units, own_forces: Units) -> int: 
    """
    Assigns a 'threat level' based on unit composition.
    You can refine logic to suit your needs.
    """
    threat_level = 0
    #TODO swap this with UNIT_DATA from Ares and change the weight calculation but filter
    threat_weights = {
        UnitTypeId.MARINE: 1.5, UnitTypeId.ZEALOT: 1.5, UnitTypeId.ZERGLING: 1.0,
        UnitTypeId.ADEPT: 2.0, UnitTypeId.STALKER: 2.0, UnitTypeId.ROACH: 2.5,
        UnitTypeId.REAPER: 1.5, UnitTypeId.MARAUDER: 3.0, UnitTypeId.SENTRY: 1.5,
        UnitTypeId.HYDRALISK: 2.5, UnitTypeId.BANELING: 3.0, UnitTypeId.HELLION: 2.0,
        UnitTypeId.HELLIONTANK: 3.0, UnitTypeId.HIGHTEMPLAR: 2.5, UnitTypeId.MUTALISK: 1.5,
        UnitTypeId.BANSHEE: 2.0, UnitTypeId.VIKING: 2.0, UnitTypeId.VIKINGFIGHTER: 2.0,
        UnitTypeId.PHOENIX: 2.0, UnitTypeId.ORACLE: 2.0, UnitTypeId.RAVEN: 2.0,
        UnitTypeId.GHOST: 2.0,
        UnitTypeId.SIEGETANK: 3.5, UnitTypeId.IMMORTAL: 3.5, UnitTypeId.CYCLONE: 3.5,
        UnitTypeId.DISRUPTOR: 3.5, UnitTypeId.COLOSSUS: 4.0, UnitTypeId.RAVAGER: 4.0,
        UnitTypeId.LURKER: 4.0, UnitTypeId.VOIDRAY: 3.0, UnitTypeId.CARRIER: 4.0,
        UnitTypeId.BATTLECRUISER: 4.5, UnitTypeId.TEMPEST: 4.0, UnitTypeId.BROODLORD: 4.5,
        UnitTypeId.ULTRALISK: 4.0, UnitTypeId.THOR: 4.5, UnitTypeId.SIEGETANKSIEGED: 4.0,
        UnitTypeId.LIBERATOR: 3.5, UnitTypeId.LIBERATORAG: 3.5, UnitTypeId.LURKERBURROWED: 4.0,
        UnitTypeId.DARKTEMPLAR: 3.0, UnitTypeId.ARCHON: 3.0, UnitTypeId.CORRUPTOR: 3.0,
        UnitTypeId.WIDOWMINE: 3.5, UnitTypeId.INFESTOR: 3.5, UnitTypeId.INFESTORBURROWED: 3.5,
        UnitTypeId.SWARMHOSTBURROWEDMP: 3.5, UnitTypeId.VIPER: 3.5, UnitTypeId.WIDOWMINEBURROWED: 3.5,
    }
    for unit in enemy_units:
        weight = threat_weights.get(unit.type_id, 1.0)
        # Multiply weight by the effective power (health + shield) scaled by 50.0, tweak if needed
        threat_level += weight * ((unit.health + unit.shield) / 50.0)

    # Density check: adjust threat level based on enemy clustering
    if enemy_units.amount > 0:
        sum_x, sum_y = 0.0, 0.0
        for unit in enemy_units:
            sum_x += unit.position.x
            sum_y += unit.position.y
        center_x = sum_x / enemy_units.amount
        center_y = sum_y / enemy_units.amount
        center = Point2((center_x, center_y))

        cluster_count = 0
        for unit in enemy_units:
            if unit.position.distance_to(center) <= 5.0:
                cluster_count += 1

        # If fewer than 3 enemy units are clustered, scale down the threat level
        if cluster_count < 3:
            threat_level *= 0.5

    # Adjust threat if our forces significantly outnumber enemy units
    if own_forces.amount > enemy_units.amount * 2:
        threat_level -= 2

    return max(round(threat_level), 0)


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
            # Begin two-threshold (hysteresis) defense logic:
            threat = assess_threat(bot, enemy_units, main_army)
            
            if bot.time < 3 * 60 and bot._used_cheese_response:
                high_threshold = 2
                low_threshold = 1
            else:
                high_threshold = 5
                low_threshold = 2
            
            if bot._under_attack:
                if threat < low_threshold:
                    bot._under_attack = False
            else:
                if threat >= high_threshold:
                    bot._under_attack = True
                    #TODO switch the role of the main army to to Defending
            
            if bot._under_attack:
                threat_position, num_units = cy_find_units_center_mass(enemy_units, 10.0)
                threat_position = Point2(threat_position)
                control_main_army(bot, main_army, threat_position, bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=15))
                if bot._one_base_reaction_completed:
                    bot.use_overcharge(main_army)


def handle_attack_toggles(bot, main_army: Units, attack_target: Point2) -> None:
    """
    Handles the attack toggles logic.
    """
    # Assess the threat level of enemy units
    #TODO measure the value of my army vs the value of the eneny to attack instead of attack supply - use get_cached_enemy_army, own_forces and enemy_forces
    enemy_threat_level = assess_threat(bot, bot.enemy_units, main_army)

    # If the army is already attacking, decide based on threat level
    if bot._commenced_attack:
        if enemy_threat_level < 5:
            # Continue attacking the original target
            control_main_army(bot, main_army, attack_target, bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=15))
        else:
            # Redirect to engage the enemy if threat is high
            enemy_center, _ = cy_find_units_center_mass(bot.enemy_units, 10.0)
            control_main_army(bot, main_army, Point2(enemy_center), bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=15))
    else:
        # Handle attack toggles logic as before
        current_supply = bot.get_total_supply(main_army)
        if current_supply <= bot._begin_attack_at_supply:
            bot._commenced_attack = False
        elif bot._commenced_attack and not bot._under_attack:
            control_main_army(bot, main_army, attack_target, bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=15))


def attack_target(bot, main_army_position: Point2) -> Point2:
    """
    Determines where our main attacking units should move toward.
    If we see an enemy structure, we target the closest one. Otherwise,
    we may fallback to expansions or the enemy start location.
    """

    if bot.enemy_structures:
        # Prioritize the closest enemy structure to the main army
        closest_structure = cy_closest_to(main_army_position, bot.enemy_structures).position
        
        # # Check if the closest structure is far away and if so, fallback to a stable target
        # if closest_structure.distance_to(main_army_position) > 50.0:
        #     return fallback_target(bot)
        
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


def regroup_army(bot, main_army: Units) -> None:
    """Regroups the main army if units are too scattered and the bot is not in a combat state.

    Conditions:
      - The average distance of units from the center of mass is greater than 10 units.
      - The bot is not currently under attack and has not commenced an attack.
    Action:
      - Command the main army to move to the natural expansion location (bot.natural_expansion).
    """
    if not main_army or main_army.amount == 0:
        return

    # Calculate the center of mass of the main army
    center, _ = cy_find_units_center_mass(main_army, 10.0)
    center = Point2(center)

    # Compute the average distance from the center
    total_distance = 0.0
    for unit in main_army:
        total_distance += unit.position.distance_to(center)
    avg_distance = total_distance / main_army.amount

    # If the army is scattered (avg distance > 10) and not in combat, regroup
    if avg_distance > 10 and not bot._under_attack and not bot._commenced_attack:
        # Command the main army to regroup by using control_main_army with the natural expansion as target
        control_main_army(bot, main_army, bot.natural_expansion.towards(bot.game_info.map_center, 2), bot.mediator.get_squads(role=UnitRole.ATTACKING, squad_radius=15))
        print("Regrouping army")