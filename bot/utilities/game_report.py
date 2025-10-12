"""
Game Report Utility
Handles presentation and formatting of end-game performance reports.
Works with PerformanceMonitor to display collected metrics.
"""


def print_end_game_report(
    performance_monitor,
    game_result,
    game_time: float,
    idle_worker_time: float,
    idle_production_time: float
) -> None:
    """
    Print streamlined end-game report using PerformanceMonitor data.
    
    Args:
        performance_monitor: PerformanceMonitor instance with collected data
        game_result: Result enum (Victory/Defeat/Tie)
        game_time: Game time in seconds
        idle_worker_time: Total idle worker time in seconds
        idle_production_time: Total idle production time in seconds
    """
    # Get combined SQ score from monitor
    sq = performance_monitor.get_current_sq()
    rating = performance_monitor.get_efficiency_rating()
    
    # Calculate combined metrics for display
    combined_income = performance_monitor.avg_income_minerals + performance_monitor.avg_income_vespene
    combined_unspent = performance_monitor.avg_unspent_minerals + performance_monitor.avg_unspent_vespene
    
    # Print clean report
    print("\n" + "="*50)
    print(f"  RESULT: {game_result}")
    print(f"  Game Time: {game_time / 60:.1f} min")
    print("="*50)
    print(f"  SPENDING QUOTIENT:  {sq:.1f}  [{rating}]")
    print("-"*50)
    print("  RESOURCES:")
    print(f"    Combined Income:  {combined_income:.0f}/min")
    print(f"    Combined Unspent: {combined_unspent:.0f}")
    print(f"      └─ Minerals: {performance_monitor.avg_unspent_minerals:.0f}m")
    print(f"      └─ Gas:      {performance_monitor.avg_unspent_vespene:.0f}g")
    print("-"*50)
    print("  EFFICIENCY:")
    print(f"    Idle Workers:     {idle_worker_time:.1f}s")
    print(f"    Idle Production:  {idle_production_time:.1f}s")
    print("="*50 + "\n")
