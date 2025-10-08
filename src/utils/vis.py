import numpy as np
import matplotlib.pyplot as plt
def plot_trajectories(truth_hist, track_hist, path):
    plt.figure()
    for k in range(truth_hist.shape[1]):
        plt.plot(truth_hist[:,k,0], truth_hist[:,k,1], linestyle='--')
    for k in range(track_hist.shape[1]):
        xs = track_hist[:,k,0]; ys = track_hist[:,k,1]
        plt.plot(xs, ys)
    plt.xlabel('x (m)'); plt.ylabel('y (m)')
    plt.title('Ground Truth (dashed) vs Tracks')
    plt.savefig(path, bbox_inches='tight'); plt.close()
