"""
Performance Monitor
Tracks bot performance metrics in real-time for both decision-making and reporting.
Can be extended to monitor additional efficiency metrics beyond Spending Quotient.
"""

import math


class PerformanceMonitor:
    """
    Monitors bot performance throughout the game.
    Tracks Spending Quotient and other efficiency metrics for real-time decision-making.
    """
    
    def __init__(self, sample_interval: int = 22, alpha: float = 0.1):
        """
        Initialize the performance monitor.
        
        Args:
            sample_interval: How often to sample (in game steps, default ~1 second)
            alpha: EMA smoothing factor (0.0-1.0, higher = more weight on recent values)
        """
        self.sample_interval = sample_interval
        self.alpha = alpha
        
        # Spending Quotient tracking
        self.avg_unspent_minerals = 0.0
        self.avg_unspent_vespene = 0.0
        self.avg_income_minerals = 0.0
        self.avg_income_vespene = 0.0
        
        # Additional performance metrics (can be extended)
        self.samples_taken = 0
        self.tracking_enabled = False  # Start tracking after early game (6 min)
    
    def update(self, iteration: int, bot) -> None:
        """
        Update performance metrics if it's time to sample.
        Only starts tracking after early game ends (game_state >= 1, after 6 minutes).
        
        Args:
            iteration: Current game iteration/step
            bot: Bot instance (to access minerals, income, etc.)
        """
        # Enable tracking once early game ends (game_state >= 1 means mid/late game)
        if not self.tracking_enabled and bot.game_state >= 1:
            self.tracking_enabled = True
        
        # Only track if enabled and it's time to sample
        if not self.tracking_enabled or iteration % self.sample_interval != 0:
            return
        
        self.samples_taken += 1
        
        # Update average unspent resources (EMA)
        self.avg_unspent_minerals = (
            self.alpha * bot.minerals + 
            (1 - self.alpha) * self.avg_unspent_minerals
        )
        self.avg_unspent_vespene = (
            self.alpha * bot.vespene + 
            (1 - self.alpha) * self.avg_unspent_vespene
        )
        
        # Update average income (EMA)
        current_income = bot.state.score.collection_rate_minerals
        current_gas_income = bot.state.score.collection_rate_vespene
        
        self.avg_income_minerals = (
            self.alpha * current_income + 
            (1 - self.alpha) * self.avg_income_minerals
        )
        self.avg_income_vespene = (
            self.alpha * current_gas_income + 
            (1 - self.alpha) * self.avg_income_vespene
        )
    
    def get_current_sq(self) -> float:
        """
        Get current Spending Quotient for real-time decision-making.
        Uses COMBINED income and unspent resources (minerals + gas together).
        
        Returns:
            Current SQ score
        """
        combined_income = self.avg_income_minerals + self.avg_income_vespene
        combined_unspent = self.avg_unspent_minerals + self.avg_unspent_vespene
        return self._calculate_sq(combined_income, combined_unspent)
    
    def is_spending_efficiently(self, threshold: float = 70.0) -> bool:
        """
        Check if bot is spending efficiently (for in-game decisions).
        
        Args:
            threshold: Minimum SQ score to be considered "efficient"
            
        Returns:
            True if SQ is above threshold
        """
        return self.get_current_sq() >= threshold
    
    def is_banking_too_much(self, threshold: float = 50.0) -> bool:
        """
        Check if bot is banking too much (needs more production).
        
        Args:
            threshold: Maximum SQ score before considered "banking"
            
        Returns:
            True if SQ is below threshold (banking problem)
        """
        return self.get_current_sq() < threshold
    
    def get_efficiency_rating(self) -> str:
        """
        Get current efficiency rating (for debugging/logging).
        
        Returns:
            Rating string (GRANDMASTER, MASTER, etc.)
        """
        sq = self.get_current_sq()
        return self._get_sq_rating(sq)
    
    def _calculate_sq(self, avg_income: float, avg_unspent: float) -> float:
        """
        Calculate Spending Quotient using formula: SQ(i,u) = 35(0.00137i - ln(u)) + 240
        
        Args:
            avg_income: Average collection rate
            avg_unspent: Average unspent resources
            
        Returns:
            SQ score
        """
        if avg_unspent <= 0:
            avg_unspent = 1  # Avoid ln(0)
        if avg_income <= 0:
            return 0
        
        return 35 * (0.00137 * avg_income - math.log(avg_unspent)) + 240
    
    def _get_sq_rating(self, sq: float) -> str:
        """Convert SQ score to skill rating."""
        if sq >= 90:
            return "GRANDMASTER"
        elif sq >= 80:
            return "MASTER"
        elif sq >= 70:
            return "DIAMOND"
        elif sq >= 60:
            return "PLATINUM"
        elif sq >= 50:
            return "GOLD"
        else:
            return "SILVER"
