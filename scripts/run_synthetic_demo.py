"""Run a synthetic sensor-fusion demo.

This script builds a synthetic marine dataset, runs a simple camera detector,
fuses camera+radar measurements, feeds them into a constant-velocity Kalman
filter tracker, and writes out basic metrics and latency reports.

Usage (PowerShell):
    python .\scripts\run_synthetic_demo.py --steps 300 --plot 1 --out outputs

Notes:
- The project root is added to sys.path so `from src.*` imports work when this
    script is executed directly (without installing the package).
- The script produces `metrics.json`, `latency_report.json`, and optionally a
    `trajectory.png` in the output folder.
"""

import argparse
import time
import os
import json
import sys
import pathlib
import numpy as np

# Ensure project root is on sys.path so `from src.*` imports work when running
# this script directly (without installing the package via pip).
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from src.datasets.synthetic import SyntheticMarineDataset
from src.perception.detector import SimpleCameraDetector
from src.tracking.kalman import CVKalman
from src.fusion.fuser import fuse_measurements
from src.utils.metrics import rmse, simple_precision_recall, write_json
from src.utils.vis import plot_trajectories

def main(steps=300,
         plot_flag=1,
         out_dir='outputs',
         ntargets=3):

    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)

    # --- Initialize dataset, detector, and tracker ---
    # Synthetic dataset with `steps` timesteps and `ntargets` targets
    ds = SyntheticMarineDataset(n_steps=steps, dt=0.1, n_targets=ntargets)
    # Simple camera detector (toy implementation)
    det = SimpleCameraDetector()
    # Constant-velocity Kalman filter tracker (tunable q/r)
    kf = CVKalman(dt=0.1, q=0.000005, r=0.001)

    # Per-target track storage (None until a track is initialized)
    tracks = [None] * ds.n_targets

    # Histories for ground-truth and estimated tracks: shape (timesteps, n_targets, 2)
    truth_hist = np.zeros((steps, ds.n_targets, 2))
    track_hist = np.zeros((steps, ds.n_targets, 2))

    # Simple latency collectors (milliseconds)
    latency = {'camera_ms': [], 'radar_ms': [], 'fuse_ms': [], 'kf_ms': [], 'total_ms': []}

    # Measurement covariance matrices for camera and radar (assumed diagonal)
    R_cam = np.diag(ds.camera_sigma**2)
    R_rad = np.diag(ds.radar_sigma**2)

    # --- Main loop: iterate through synthetic dataset stream ---
    for t, truth, cam_raw, rad_raw in ds.stream():
        t0 = time.time()  # top-of-loop timestamp for total latency

        # Save ground-truth positions for metrics
        truth_hist[t] = truth

        # ---- Camera detection ----
        t_cam0 = time.time()
        cam_det = det.infer_points(cam_raw)  # list of detections or None per target
        t_cam1 = time.time()

        # Convert detection dicts to position tuples or None
        cam_meas = [c['xy'] if c is not None else None for c in cam_det]

        # ---- Radar measurements (assumed available directly from dataset) ----
        t_rad0 = time.time()
        rad_meas = rad_raw
        t_rad1 = time.time()

        # ---- Fuse camera + radar measurements per target ----
        fused = []
        t_fuse0 = time.time()
        for i in range(ds.n_targets):
            # `fuse_measurements` returns (z, R) where z is fused position (or None)
            # and R is the fused covariance matrix for that measurement.
            z, R = fuse_measurements([cam_meas[i], rad_meas[i]], [R_cam, R_rad])
            fused.append((z, R))
        t_fuse1 = time.time()

        # ---- Tracking: Kalman init/predict/update per target ----
        t_kf0 = time.time()
        for i in range(ds.n_targets):
            z, R = fused[i]

            # If this target doesn't have an initialized track yet, initialize if
            # we have a measurement `z` for it.
            if tracks[i] is None:
                if z is not None:
                    # Initialize state and covariance (with optional init velocity)
                    x, P = kf.init_track(z, init_vel=(0.0, 0.0), p0=5.0)
                    tracks[i] = (x, P)
                else:
                    # No measurement and no existing track => nothing to do
                    continue
            else:
                x, P = tracks[i]

            # Prediction step
            x_pred, P_pred = kf.predict(x, P)

            # Update if we have a measurement; temporarily set kf.R to fused R
            if z is not None:
                R_old = kf.R.copy()
                kf.R = R
                x_upd, P_upd, _ = kf.update(x_pred, P_pred, z)
                kf.R = R_old
                tracks[i] = (x_upd, P_upd)
            else:
                # No measurement this timestep => use prediction as new state
                tracks[i] = (x_pred, P_pred)

            # Record the current estimated position for this target/time
            track_hist[t, i, 0] = tracks[i][0][0]
            track_hist[t, i, 1] = tracks[i][0][1]
        t_kf1 = time.time()

        # Collect latency numbers (convert seconds->milliseconds)
        latency['camera_ms'].append(1000 * (t_cam1 - t_cam0))
        latency['radar_ms'].append(1000 * (t_rad1 - t_rad0))
        latency['fuse_ms'].append(1000 * (t_fuse1 - t_fuse0))
        latency['kf_ms'].append(1000 * (t_kf1 - t_kf0))
        latency['total_ms'].append(1000 * (time.time() - t0))

    # --- Post-processing: detection metrics and RMSE ---
    det_metrics = simple_precision_recall(
        # flatten ground-truth and detection lists for the metric helper
        gt_visible=[tuple(truth_hist[i, j]) for i in range(steps) for j in range(ds.n_targets)],
        det_list=[tuple(track_hist[i, j]) if np.any(track_hist[i, j] != 0) else None
                  for i in range(steps) for j in range(ds.n_targets)],
        tol=3.0)

    # Compute a simple RMSE across all stored positions and write metrics
    write_json(os.path.join(out_dir, 'metrics.json'),
               {'rmse_position': float(np.sqrt(((truth_hist - track_hist) ** 2).mean())), **det_metrics})

    # Summarize latency statistics (mean and 95th percentile)
    lat_summary = {k: {'mean_ms': float(np.mean(v)), 'p95_ms': float(np.percentile(v, 95))}
                   for k, v in latency.items()}
    with open(os.path.join(out_dir, 'latency_report.json'), 'w') as f:
        json.dump({'summary': lat_summary}, f, indent=2)

    # Optionally generate a trajectory plot (truth vs estimated)
    if plot_flag:
        plot_trajectories(truth_hist, track_hist, os.path.join(out_dir, 'trajectory.png'))

    print('Done. See outputs/.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Run synthetic sensor-fusion demo')
    ap.add_argument('--steps', type=int, default=300, help='number of timesteps to simulate')
    ap.add_argument('--plot', type=int, default=1, help='whether to generate a trajectory plot (1/0)')
    ap.add_argument('--ntargets', type=int, default=3, help='number of targets in the simulation')
    ap.add_argument('--out', type=str, default='outputs', help='output directory for metrics and plots')
    args = ap.parse_args()
    main(steps=args.steps, plot_flag=args.plot, out_dir=args.out, ntargets=args.ntargets)
