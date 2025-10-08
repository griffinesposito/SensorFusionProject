"""Simple top-down grid map environment with random rectangular obstacles.

Provides a GridMap class that holds an m-by-n grid and can populate it with N
random axis-aligned rectangular obstacles of random sizes. Includes a
convenience plotting method that draws open cells white and blocked cells
black.

Example:
    from src.utils.map_env import GridMap
    g = GridMap(50, 80, seed=0)
    g.populate_obstacles(20, min_size=(1,1), max_size=(6,6))
    g.plot(save_path='map.png', show=False)
"""

from typing import Tuple
import numpy as np


class GridMap:
    """Top-down 2D grid map with random rectangular obstacles.

    Grid cells are stored as a numpy array with values:
      0 -> open (free)
      1 -> blocked (obstacle)

    Args:
        m (int): number of rows (height)
        n (int): number of columns (width)
        seed (int|None): optional random seed for reproducibility
    """

    def __init__(self, m: int, n: int, seed: int | None = None):
        if m <= 0 or n <= 0:
            raise ValueError('m and n must be positive integers')
        self.m = int(m)
        self.n = int(n)
        self.grid = np.zeros((self.m, self.n), dtype=np.uint8)
        self._rng = np.random.default_rng(seed)
        # Optional start and end cells (row, col) assigned after obstacles are
        # placed. They are None until assigned via `assign_start_end`.
        self.start_location: tuple | None = None
        self.end_location: tuple | None = None

    def reset(self) -> None:
        """Clear the map (make all cells open)."""
        self.grid.fill(0)

    def populate_obstacles(self,
                           N: int,
                           min_size: Tuple[int, int] = (1, 1),
                           max_size: Tuple[int, int] = (5, 5),
                           max_attempts_per_obstacle: int = 100) -> int:
        """Place up to N random rectangular obstacles into the grid.

        Obstacles are axis-aligned rectangles with integer sizes sampled
        uniformly between `min_size` and `max_size` (inclusive). Placement is
        attempted up to `max_attempts_per_obstacle` times per obstacle; if a
        non-overlapping position cannot be found the obstacle is skipped.

        Returns:
            int: number of obstacles successfully placed.
        """
        if N <= 0:
            return 0

        min_h, min_w = int(min_size[0]), int(min_size[1])
        max_h, max_w = int(max_size[0]), int(max_size[1])
        if min_h <= 0 or min_w <= 0 or max_h < min_h or max_w < min_w:
            raise ValueError('invalid min_size / max_size')

        placed = 0
        for _ in range(N):
            placed_this = False
            for _attempt in range(max_attempts_per_obstacle):
                h = self._rng.integers(min_h, max_h + 1)
                w = self._rng.integers(min_w, max_w + 1)

                if h > self.m or w > self.n:
                    # obstacle too large for the grid
                    continue

                # choose top-left corner (row, col) so the rectangle fits
                r = self._rng.integers(0, self.m - h + 1)
                c = self._rng.integers(0, self.n - w + 1)

                # check for overlap with existing obstacles
                if np.any(self.grid[r:r + h, c:c + w] != 0):
                    continue

                # place obstacle (mark cells as blocked -> 1)
                self.grid[r:r + h, c:c + w] = 1
                placed += 1
                placed_this = True
                break

            if not placed_this:
                # failed to place this obstacle after many attempts; skip it
                continue

        return placed

    def assign_start_end(self, start: Tuple[int, int] | None = None,
                         end: Tuple[int, int] | None = None,
                         distinct: bool = True) -> Tuple[Tuple[int, int] | None, Tuple[int, int] | None]:
        """Assign start and end locations to random free cells.

        If `start` or `end` is provided it must be a (row, col) tuple pointing
        to an open cell (value 0). If None the location is chosen randomly
        among currently open cells. If `distinct` is True the end location
        will be chosen different from the start location.

        Returns:
            (start, end) tuple of assigned coordinates or None if assignment
            failed (e.g., no free cells available).
        """
        free_cells = np.argwhere(self.grid == 0)

        if start is not None:
            sr, sc = int(start[0]), int(start[1])
            if self.grid[sr, sc] != 0:
                raise ValueError('start location must be an open cell')
            self.start_location = (sr, sc)
        else:
            if free_cells.size == 0:
                self.start_location = None
            else:
                idx = self._rng.integers(0, len(free_cells))
                r, c = free_cells[idx]
                self.start_location = (int(r), int(c))

        # Recompute free cells if we chose a start (so end won't pick it)
        free_cells = np.argwhere(self.grid == 0)
        if self.start_location is not None and distinct:
            # filter out the start cell
            mask = ~((free_cells[:, 0] == self.start_location[0]) & (free_cells[:, 1] == self.start_location[1]))
            free_cells = free_cells[mask]

        if end is not None:
            er, ec = int(end[0]), int(end[1])
            if self.grid[er, ec] != 0:
                raise ValueError('end location must be an open cell')
            if distinct and self.start_location is not None and (er, ec) == self.start_location:
                raise ValueError('end must be distinct from start')
            self.end_location = (er, ec)
        else:
            if free_cells.size == 0:
                self.end_location = None
            else:
                idx = self._rng.integers(0, len(free_cells))
                r, c = free_cells[idx]
                self.end_location = (int(r), int(c))

        return self.start_location, self.end_location

    def plot(self, ax=None, figsize=(6, 6), cmap: str = 'gray_r',
             save_path: str | None = None, show: bool = True, path: list | None = None):
        """Plot the grid map.

        Open cells are white and blocked cells are black by default (using the
        'gray_r' colormap). If `save_path` is provided the figure is saved to
        that path. If `show` is False the figure is closed after saving.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
            created_fig = True
        else:
            created_fig = False

        # origin='lower' makes the (0,0) index appear at the bottom-left, which
        # is often more intuitive for top-down maps.
        im = ax.imshow(self.grid, cmap=cmap, origin='lower', interpolation='nearest')
        # If start/end locations exist, overlay circle markers so they remain
        # visible regardless of grid cell size. Use scatter with s (area in
        # points^2) that scales inversely with the larger grid dimension.
        if self.start_location is not None or self.end_location is not None:
            # size scaling: shrink marker as grid gets larger
            s = int(max(40, 2000 / max(1, max(self.m, self.n))))
        if self.start_location is not None:
            sr, sc = self.start_location
            # scatter uses x=col, y=row coordinates (because imshow uses (col,row))
            ax.scatter([sc], [sr], c='green', s=s, marker='o', edgecolor='black', zorder=3)
        if self.end_location is not None:
            er, ec = self.end_location
            ax.scatter([ec], [er], c='blue', s=s, marker='o', edgecolor='black', zorder=3)

        # If a path is provided, plot it as a red line connecting cell centers
        if path is not None and len(path) > 0:
            # Convert (r,c) pairs to x,y for plotting (x=col, y=row)
            xs = [c for (r, c) in path]
            ys = [r for (r, c) in path]
            ax.plot(xs, ys, color='red', linewidth=2, zorder=4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'GridMap {self.m}x{self.n} (blocked cells black)')

        if save_path is not None:
            plt.savefig(save_path, bbox_inches='tight')

        if show:
            plt.show()
        elif created_fig:
            # close the figure if we created it and were asked not to show it
            plt.close(fig)

        return ax

    def get_grid(self) -> np.ndarray:
        """Return a copy of the internal grid array (0=open, 1=blocked)."""
        return self.grid.copy()

    def __repr__(self) -> str:
        return f'<GridMap {self.m}x{self.n}, blocked={int(self.grid.sum())}>'
