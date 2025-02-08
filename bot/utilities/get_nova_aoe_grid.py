
"""Module to generate a grid for Disruptor Nova AOE targeting.

This module provides a function to create and initialize a grid that can be used to calculate
influence values for optimizing Disruptor Nova target selection. The grid represents the
battlefield as a 2D matrix of cells, where each cell can later be updated with influence
calculations based on enemy and friendly unit positions.
"""
import math

from map_analyzer import MapData
import numpy as np
from sc2.position import Point2


def get_nova_aoe_grid(grid_width: int, grid_height: int, cell_size: float) -> list:
    """
    Generate a 2D grid for Disruptor Nova AOE influence calculations.

    Args:
        grid_width (int): Number of cells horizontally.
        grid_height (int): Number of cells vertically.
        cell_size (float): The size of each cell (in game units).

    Returns:
        list: A 2D list (grid) initialized with zeros, representing default influence values.
    """
    grid = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
    return grid


# Placeholder for future grid update logic

def update_grid_influence(grid: list, enemy_units: list, friendly_units: list, cell_size: float) -> list:
    """
    Update the grid influence values based on the positions of enemy and friendly units.

    Uses a linear decay function to compute influence from units within a specified radius.
    Each enemy unit adds positive influence and each friendly unit subtracts influence.

    Args:
        grid (list): The current grid to update.
        enemy_units (list): List of enemy unit positions as tuples (x, y).
        friendly_units (list): List of friendly unit positions as tuples (x, y).
        cell_size (float): The size of each grid cell (in game units).

    Returns:
        list: The updated grid with recalculated influence values.
    """
    influence_radius = cell_size * 5  # radius within which units influence the grid
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    for i in range(height):
        for j in range(width):
            cell_center_x = j * cell_size + cell_size / 2
            cell_center_y = i * cell_size + cell_size / 2
            influence_value = 0.0
            # Add influence from enemy units
            for enemy in enemy_units:
                # Assuming enemy is a tuple (x, y)
                ex, ey = enemy
                dx = ex - cell_center_x
                dy = ey - cell_center_y
                distance = math.sqrt(dx * dx + dy * dy)
                if distance < influence_radius:
                    influence_value += (1 - distance / influence_radius)
            # Subtract influence from friendly units
            for friend in friendly_units:
                fx, fy = friend
                dx = fx - cell_center_x
                dy = fy - cell_center_y
                distance = math.sqrt(dx * dx + dy * dy)
                if distance < influence_radius:
                    influence_value -= (1 - distance / influence_radius)
            grid[i][j] = influence_value
    return grid
