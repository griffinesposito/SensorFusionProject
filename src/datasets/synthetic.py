import numpy as np

class SyntheticMarineDataset:
    def __init__(self, n_steps=300, dt=0.1, n_targets=3, seed=7):
        rng = np.random.default_rng(seed)
        self.dt = dt
        self.n_steps = n_steps
        self.n_targets = n_targets
        self.x0 = rng.uniform([-80,-20], [20,20], size=(n_targets,2))
        self.v  = rng.uniform([0.3,-0.2], [1.0,0.6], size=(n_targets,2))
        self.camera_sigma = np.array([0.8, 0.8])
        self.radar_sigma  = np.array([1.5, 1.5])
        self.p_miss_camera = 0.1
        self.p_miss_radar  = 0.12
        self.rng = rng

    def ground_truth(self, t):
        return self.x0 + self.v * (t * self.dt)

    def camera_detect(self, truth):
        dets = []
        for (x, y) in truth:
            if self.rng.random() < self.p_miss_camera:
                dets.append(None)
            else:
                noise = self.rng.normal(0, self.camera_sigma, size=2)
                dets.append(np.array([x, y]) + noise)
        return dets

    def radar_detect(self, truth):
        dets = []
        for (x, y) in truth:
            if self.rng.random() < self.p_miss_radar:
                dets.append(None)
            else:
                noise = self.rng.normal(0, self.radar_sigma, size=2)
                dets.append(np.array([x, y]) + noise)
        return dets

    def stream(self):
        for t in range(self.n_steps):
            truth = self.ground_truth(t)
            cam = self.camera_detect(truth)
            rad = self.radar_detect(truth)
            yield t, truth, cam, rad
