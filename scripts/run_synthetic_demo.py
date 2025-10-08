import argparse, time, os, json
import numpy as np
from src.datasets.synthetic import SyntheticMarineDataset
from src.perception.detector import SimpleCameraDetector
from src.tracking.kalman import CVKalman
from src.fusion.fuser import fuse_measurements
from src.utils.metrics import rmse, simple_precision_recall, write_json
from src.utils.vis import plot_trajectories
def main(steps=300, plot_flag=1, out_dir='outputs'):
    os.makedirs(out_dir, exist_ok=True)
    ds = SyntheticMarineDataset(n_steps=steps, dt=0.1, n_targets=3)
    det = SimpleCameraDetector(); kf = CVKalman(dt=0.1, q=0.05, r=1.0)
    tracks=[None]*ds.n_targets
    truth_hist=np.zeros((steps, ds.n_targets, 2)); track_hist=np.zeros((steps, ds.n_targets, 2))
    latency={'camera_ms':[], 'radar_ms':[], 'fuse_ms':[], 'kf_ms':[], 'total_ms':[]}
    R_cam=np.diag(ds.camera_sigma**2); R_rad=np.diag(ds.radar_sigma**2)
    for t, truth, cam_raw, rad_raw in ds.stream():
        t0=time.time(); truth_hist[t]=truth
        t_cam0=time.time(); cam_det=det.infer_points(cam_raw); t_cam1=time.time()
        cam_meas=[c['xy'] if c is not None else None for c in cam_det]
        t_rad0=time.time(); rad_meas=rad_raw; t_rad1=time.time()
        fused=[]; t_fuse0=time.time()
        for i in range(ds.n_targets):
            z,R=fuse_measurements([cam_meas[i], rad_meas[i]],[R_cam,R_rad]); fused.append((z,R))
        t_fuse1=time.time()
        t_kf0=time.time()
        for i in range(ds.n_targets):
            z,R=fused[i]
            if tracks[i] is None:
                if z is not None:
                    x,P=kf.init_track(z, init_vel=(0.0,0.0), p0=5.0); tracks[i]=(x,P)
                else: continue
            else: x,P=tracks[i]
            x_pred,P_pred=kf.predict(x,P)
            if z is not None:
                R_old=kf.R.copy(); kf.R=R
                x_upd,P_upd,_=kf.update(x_pred,P_pred,z); kf.R=R_old; tracks[i]=(x_upd,P_upd)
            else: tracks[i]=(x_pred,P_pred)
            track_hist[t,i,0]=tracks[i][0][0]; track_hist[t,i,1]=tracks[i][0][1]
        t_kf1=time.time()
        latency['camera_ms'].append(1000*(t_cam1-t_cam0))
        latency['radar_ms'].append(1000*(t_rad1-t_rad0))
        latency['fuse_ms'].append(1000*(t_fuse1-t_fuse0))
        latency['kf_ms'].append(1000*(t_kf1-t_kf0))
        latency['total_ms'].append(1000*(time.time()-t0))
    det_metrics=simple_precision_recall(
        gt_visible=[tuple(truth_hist[i,j]) for i in range(steps) for j in range(ds.n_targets)],
        det_list=[tuple(track_hist[i,j]) if np.any(track_hist[i,j]!=0) else None for i in range(steps) for j in range(ds.n_targets)],
        tol=3.0)
    write_json(os.path.join(out_dir,'metrics.json'),{'rmse_position': float(np.sqrt(((truth_hist-track_hist)**2).mean())), **det_metrics})
    lat_summary={k:{'mean_ms':float(np.mean(v)),'p95_ms':float(np.percentile(v,95))} for k,v in latency.items()}
    with open(os.path.join(out_dir,'latency_report.json'),'w') as f: json.dump({'summary':lat_summary}, f, indent=2)
    if plot_flag: plot_trajectories(truth_hist, track_hist, os.path.join(out_dir,'trajectory.png'))
    print('Done. See outputs/.')
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--steps',type=int,default=300)
    ap.add_argument('--plot',type=int,default=1); ap.add_argument('--out',type=str,default='outputs')
    args=ap.parse_args(); main(steps=args.steps, plot_flag=args.plot, out_dir=args.out)
