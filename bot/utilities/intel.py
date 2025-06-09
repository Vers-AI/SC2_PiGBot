"""
Utility functions for gathering intelligence about the enemy.
"""
from sc2.data import Race
from sc2.ids.unit_typeid import UnitTypeId as UnitID


def get_enemy_cannon_rushed(bot, detection_radius: float = 25.0) -> bool:
    """Check if the enemy is cannon rushing.
    
    Args:
        bot: The bot instance
        detection_radius: Distance in game units to check around each base (default: 25.0)
        
    Returns:
        bool: True if cannon rush is detected, False otherwise
    """
    # Only check against Protoss
    if bot.enemy_race != Race.Protoss:
        return False
        
    # Only check in early game (before 3 minutes)
    if bot.time > 180.0:  # 3 minutes
        return False
    
    try:
        # Get main base and natural expansion positions
        main_base = bot.start_location
        natural_expansion = bot.mediator.get_own_nat  # This gets the natural expansion position
        
        # Look for enemy pylons and cannons near our bases
        enemy_structures = bot.enemy_structures.filter(
            lambda s: s.type_id in {UnitID.PYLON, UnitID.PHOTONCANNON} and 
                     (s.distance_to(main_base) < detection_radius or 
                      s.distance_to(natural_expansion) < detection_radius)
        )
        
        # Count pylons and cannons
        pylon_count = sum(1 for s in enemy_structures if s.type_id == UnitID.PYLON)
        cannon_count = sum(1 for s in enemy_structures if s.type_id == UnitID.PHOTONCANNON)
        
        # If we see cannons or multiple pylons near our bases, it's likely a cannon rush
        if cannon_count > 0 or pylon_count > 1:
            print(f"Cannon rush detected with {cannon_count} cannons and {pylon_count} pylons within {detection_radius} units of our bases")
            return True
            
        return False
        
    except Exception as e:
        print(f"Error in cannon rush detection: {e}")
        return False
