"""Run a small demo that creates and plots a GridMap.

This script is a convenience for quickly generating a test occupancy grid and
saving a PNG. It mirrors the functionality described in
`src/utils/map_env.py`.
"""
import argparse
import os
import sys
import pathlib

# Ensure project root is on sys.path so `from src.*` imports work when running
# this script directly (without installing the package via pip).
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
from src.utils.map_env import GridMap
from src.navigation.drpg import DRPG
from src.navigation.astar import astar_search


def main():
    ap = argparse.ArgumentParser(description='Generate and plot a test GridMap')
    ap.add_argument('--m', type=int, default=600, help='rows (height)')
    ap.add_argument('--n', type=int, default=600, help='cols (width)')
    ap.add_argument('--N', type=int, default=250, help='number of obstacles to place')
    ap.add_argument('--min_h', type=int, default=10, help='minimum obstacle height')
    ap.add_argument('--min_w', type=int, default=10, help='minimum obstacle width')
    ap.add_argument('--max_h', type=int, default=50, help='maximum obstacle height')
    ap.add_argument('--max_w', type=int, default=50, help='maximum obstacle width')
    ap.add_argument('--seed', type=int, default=None, help='random seed')
    ap.add_argument('--out', type=str, default='outputs', help='output directory')
    ap.add_argument('--fname', type=str, default='map_demo.png', help='output filename')
    ap.add_argument('--show', type=int, default=0, help='show plot interactively (1)')
    ap.add_argument('--episodes', type=int, default=2000, help='training episodes for DRPG')
    ap.add_argument('--max_steps', type=int, default=2000, help='max steps when extracting path')
    ap.add_argument('--planner', type=str, default='drpg', help='planner to use: drpg or astar')
    ap.add_argument('--diag', action='store_true', help='allow diagonal movement for astar')

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    gm = GridMap(args.m, args.n, seed=args.seed)
    placed = gm.populate_obstacles(args.N, min_size=(args.min_h, args.min_w),
                                   max_size=(args.max_h, args.max_w))
    print(f'placed {placed}/{args.N} obstacles')

    # assign random start and end locations on free cells
    start, end = gm.assign_start_end()
    print(f'start={start}, end={end}')

    path = None
    if args.planner.lower() == 'drpg':
        # Train a DRPG agent to find a path from start to end
        drpg = DRPG(gm)
        try:
            drpg.train(episodes=args.episodes, verbose=False)
            path = drpg.extract_path(max_steps=args.max_steps)
            print('extracted path length:', len(path) if path is not None else None)
        except Exception as e:
            print('DRPG training/extract failed:', e)
            path = None
    elif args.planner.lower() == 'astar':
        try:
            path = astar_search(gm, start, end, allow_diagonal=args.diag)
            print('astar path length:', len(path) if path is not None else None)
        except Exception as e:
            print('A* failed:', e)
            path = None
    else:
        print('unknown planner:', args.planner)

    save_path = os.path.join(args.out, args.fname)
    gm.plot(figsize=(8, 6), save_path=save_path, show=bool(args.show), path=path)
    print(f'wrote {save_path}')


if __name__ == '__main__':
    main()
