import numpy as np
class SimpleCameraDetector:
    def __init__(self, sigma=(0.8,0.8)):
        self.sigma = np.array(sigma)
    def infer_points(self, camera_points):
        outputs = []
        for p in camera_points:
            outputs.append({'xy': p, 'score': 0.9} if p is not None else None)
        return outputs
