"""Dynamic Real-time Path Guidance (DRPG) - simple Q-learning agent for grid maps.

This is a small, self-contained Q-learning implementation that treats the
grid as a discrete MDP. It's intended as a demo: with per-step penalty and a
large goal reward, the learned policy approximates a shortest-path policy on
deterministic grids.
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np


class DRPG:
    """Q-learning based path planner on a GridMap.

    The agent uses actions [0:up, 1:right, 2:down, 3:left]. Attempting to
    move into an obstacle or outside the grid results in staying in place and
    receiving an extra penalty.
    """

    ACTIONS = [( -1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(self, gridmap, alpha: float = 0.1, gamma: float = 0.99,
                 epsilon: float = 0.2, epsilon_decay: float = 0.999, min_epsilon: float = 0.01,
                 hit_obstacle_penalty: float = -5.0, step_penalty: float = -1.0,
                 goal_reward: float = 100.0,
                 shaping_scale: float = 2.0):
        self.gm = gridmap
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.hit_obstacle_penalty = hit_obstacle_penalty
        self.step_penalty = step_penalty
        self.goal_reward = goal_reward

        self.n_states = self.gm.m * self.gm.n
        self.n_actions = len(self.ACTIONS)
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=float)
        self._rng = np.random.default_rng()

        # shaping scale controls how much we reward progress toward goal
        self.shaping_scale = float(shaping_scale)

    def _rc_to_state(self, r: int, c: int) -> int:
        return int(r) * self.gm.n + int(c)

    def _state_to_rc(self, s: int) -> Tuple[int, int]:
        return divmod(int(s), self.gm.n)

    def _is_goal(self, s: int) -> bool:
        if self.gm.end_location is None:
            return False
        er, ec = self.gm.end_location
        return s == self._rc_to_state(er, ec)

    def _heuristic_state(self, s: int) -> float:
        """Manhattan distance heuristic from state s to goal (if goal exists)."""
        if self.gm.end_location is None:
            return 0.0
        r, c = self._state_to_rc(s)
        er, ec = self.gm.end_location
        return float(abs(r - er) + abs(c - ec))

    def _initialize_q_with_heuristic(self) -> None:
        """Initialize Q-values using negative heuristic of successor states.

        For each state-action pair we set Q[s,a] = -h(s') where s' is the
        resulting state after taking action a. This biases greedy actions
        toward the goal and gives learning a strong prior.
        """
        if self.gm.end_location is None:
            return
        for s in range(self.n_states):
            r, c = self._state_to_rc(s)
            for a_idx, (dr, dc) in enumerate(self.ACTIONS):
                nr, nc = r + dr, c + dc
                # if move is invalid or blocked, keep a large negative value
                if not (0 <= nr < self.gm.m and 0 <= nc < self.gm.n) or self.gm.grid[nr, nc] != 0:
                    self.Q[s, a_idx] = -1e3
                else:
                    ns = self._rc_to_state(nr, nc)
                    self.Q[s, a_idx] = -self._heuristic_state(ns)

    def _step_from_state_action(self, s: int, a: int) -> Tuple[int, float, bool]:
        r, c = self._state_to_rc(s)
        dr, dc = self.ACTIONS[a]
        nr, nc = r + dr, c + dc

        # Check bounds
        if not (0 <= nr < self.gm.m and 0 <= nc < self.gm.n):
            # invalid move: stay in place and apply obstacle penalty
            return s, self.hit_obstacle_penalty, False

        # Check obstacle
        if self.gm.grid[nr, nc] != 0:
            # blocked cell: stay and penalize
            return s, self.hit_obstacle_penalty, False

        next_s = self._rc_to_state(nr, nc)
        # Check goal
        if self._is_goal(next_s):
            return next_s, self.goal_reward, True

        return next_s, self.step_penalty, False

    def _choose_action(self, s: int) -> int:
        # epsilon-greedy
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return int(np.argmax(self.Q[s]))

    def train(self, episodes: int = 1000, max_steps_per_episode: int = 1000, verbose: bool = False):
        """Train the Q-table for a number of episodes starting at the map's
        start_location.
        """
        if self.gm.start_location is None or self.gm.end_location is None:
            raise RuntimeError('GridMap must have start_location and end_location assigned')

        sr, sc = self.gm.start_location
        start_s = self._rc_to_state(sr, sc)

        # Initialize Q with heuristic prior to speed up learning
        self._initialize_q_with_heuristic()

        for ep in range(episodes):
            s = start_s
            done = False
            for step in range(max_steps_per_episode):
                a = self._choose_action(s)
                # compute shaped reward: base reward plus progress toward goal
                old_dist = self._heuristic_state(s)
                next_s, reward, done = self._step_from_state_action(s, a)
                new_dist = self._heuristic_state(next_s)
                # positive bonus for reducing distance to goal
                shaping = self.shaping_scale * (old_dist - new_dist)
                reward = reward + shaping

                # Q update (standard Q-learning)
                td_target = reward + (0 if done else self.gamma * np.max(self.Q[next_s]))
                td_error = td_target - self.Q[s, a]
                self.Q[s, a] += self.alpha * td_error

                s = next_s
                if done:
                    break

            # decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            if verbose and (ep % max(100, episodes // 10) == 0):
                print(f'Episode {ep}/{episodes}, epsilon={self.epsilon:.4f}')

    def extract_path(self, max_steps: int = 10000) -> Optional[List[Tuple[int, int]]]:
        """Extract a path by following the greedy policy from start to goal.

        Returns a list of (row, col) tuples or None if no path found.
        """
        if self.gm.start_location is None or self.gm.end_location is None:
            return None
        sr, sc = self.gm.start_location
        s = self._rc_to_state(sr, sc)
        visited = set()
        path = [(sr, sc)]

        for _ in range(max_steps):
            a = int(np.argmax(self.Q[s]))
            next_s, reward, done = self._step_from_state_action(s, a)
            if next_s == s:
                # stuck; no progress
                return None
            r, c = self._state_to_rc(next_s)
            path.append((r, c))
            if (r, c) == self.gm.end_location:
                return path
            if (r, c) in visited:
                # cycle detected
                return None
            visited.add((r, c))
            s = next_s

        return None
