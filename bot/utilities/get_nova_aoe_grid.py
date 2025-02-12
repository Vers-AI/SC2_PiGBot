"""Module to generate a grid for Disruptor Nova AOE targeting.

This module provides functions to create and initialize a grid that can be used to calculate
influence values for optimizing Disruptor Nova target selection. The grid represents the
battlefield as a 2D matrix of cells, where each cell can later be updated with influence
calculations based on enemy and friendly unit positions.
"""
import math

from map_analyzer import MapData
import numpy as np
from sc2.position import Point2


def get_nova_aoe_grid(map_data: MapData) -> np.ndarray:
    """
    Generate a grid for Disruptor Nova using MapData.

    This function retrieves a grid using MapData.get_pyastar_grid(), then adjusts the grid by replacing
    values: non-pathable cells (0) are set to -10000 and pathable cells (1) are set to 0, mimicking the behavior
    of pre-filled pathing data with developer-defined costs.

    Args:
        map_data (MapData): The map analyzer object providing map information.

    Returns:
        np.ndarray: The modified grid ready for further influence updates.
    """
    grid = map_data.get_pyastar_grid()
    # Replace non-pathable cells (0) with -10000 and pathable cells (1) with 0
    grid[grid == 0] = -10000
    grid[grid == 1] = 0
    return grid


# Placeholder for future grid update logic


def apply_influence_in_radius(grid: np.ndarray, center: tuple, radius: float, influence: float, map_data=None) -> np.ndarray:
    """
    Apply influence to grid cells within a given radius around a center point using a linear decay function.

    If map_data is provided, use its add_cost method to apply the influence directly to the grid.

    Args:
        grid (np.ndarray): The grid to update.
        center (tuple): The (row, col) coordinates of the center point in the grid.
        radius (float): The radius (in grid cell units) within which to apply influence.
        influence (float): The maximum influence value to apply at the center; it decays linearly to 0 at the edge.
        map_data: Optional. An instance with an add_cost method to apply custom cost updates.

    Returns:
        np.ndarray: The updated grid with influence applied.
    """
    if map_data is not None:
        # Use map_data.add_cost to add influence. Use safe=False to allow negative influence if needed.
        grid = map_data.add_cost(position=center, radius=radius, grid=grid, weight=influence, safe=False)
        return grid
    else:
        rows, cols = grid.shape
        center_row, center_col = center
        # Determine the bounding box (clipping to grid boundaries)
        row_start = max(0, int(center_row - radius))
        row_end = min(rows, int(center_row + radius) + 1)
        col_start = max(0, int(center_col - radius))
        col_end = min(cols, int(center_col + radius) + 1)

        for i in range(row_start, row_end):
            for j in range(col_start, col_end):
                # Calculate Euclidean distance from center
                dist = math.sqrt((i - center_row)**2 + (j - center_col)**2)
                if dist < radius:
                    # Linear decay: influence decreases linearly from center to radius
                    decay_factor = 1 - (dist / radius)
                    grid[i][j] += influence * decay_factor
        return grid
