"""A* planner for GridMap occupancy grids.

Returns a shortest path (in number of steps) from start to goal using a
Manhattan-distance heuristic. The grid is treated as 4-connected (up/right/down/left).
"""
from typing import List, Tuple, Optional
import heapq


def astar_search(grid: 'GridMap', start: Tuple[int, int], goal: Tuple[int, int], allow_diagonal: bool = False) -> Optional[List[Tuple[int, int]]]:
    """A* search on a GridMap.

    If `allow_diagonal` is True, the planner uses 8-connected movement where
    diagonal moves have cost sqrt(2) and the heuristic is the octile distance.
    Otherwise it uses 4-connected movement with unit-cost moves and Manhattan
    heuristic.
    """
    import math

    m, n = grid.m, grid.n

    def in_bounds(r, c):
        return 0 <= r < m and 0 <= c < n

    # Movement offsets and their costs
    if allow_diagonal:
        moves = [(-1,0,1.0),(0,1,1.0),(1,0,1.0),(0,-1,1.0),(-1,-1,math.sqrt(2)),(-1,1,math.sqrt(2)),(1,1,math.sqrt(2)),(1,-1,math.sqrt(2))]
    else:
        moves = [(-1,0,1.0),(0,1,1.0),(1,0,1.0),(0,-1,1.0)]

    def neighbors_with_cost(r, c):
        for dr, dc, cost in moves:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc) and grid.grid[nr, nc] == 0:
                yield (nr, nc, cost)

    def h(a, b):
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if allow_diagonal:
            # octile distance heuristic: D=1, D2=sqrt(2)
            D = 1.0
            D2 = math.sqrt(2.0)
            return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
        else:
            # Manhattan
            return dx + dy

    start = tuple(start); goal = tuple(goal)
    if not in_bounds(*start) or not in_bounds(*goal):
        return None
    if grid.grid[start[0], start[1]] != 0 or grid.grid[goal[0], goal[1]] != 0:
        return None

    open_set = []
    heapq.heappush(open_set, (h(start, goal), 0.0, start))
    came_from = {start: None}
    gscore = {start: 0.0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        if current == goal:
            # reconstruct path
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return list(reversed(path))

        # expand neighbors with costs
        for nr, nc, move_cost in neighbors_with_cost(*current):
            nb = (nr, nc)
            tentative_g = gscore[current] + move_cost
            if nb not in gscore or tentative_g < gscore[nb]:
                came_from[nb] = current
                gscore[nb] = tentative_g
                f = tentative_g + h(nb, goal)
                heapq.heappush(open_set, (f, tentative_g, nb))

    return None
