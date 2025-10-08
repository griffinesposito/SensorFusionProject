import numpy as np
class CVKalman:
    def __init__(self, dt=0.1, q=0.1, r=1.0):
        self.dt = dt
        self.F = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]])
        self.H = np.array([[1,0,0,0],[0,1,0,0]])
        self.Q = q * np.eye(4)
        self.R = r * np.eye(2)
    def init_track(self, xy, init_vel=(0.0,0.0), p0=10.0):
        x = np.array([xy[0], xy[1], init_vel[0], init_vel[1]], dtype=float)
        P = p0 * np.eye(4); return x, P
    def predict(self, x, P):
        x_pred = self.F @ x; P_pred = self.F @ P @ self.F.T + self.Q; return x_pred, P_pred
    def update(self, x_pred, P_pred, z):
        y = z - (self.H @ x_pred); S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        x_new = x_pred + K @ y; P_new = (np.eye(4) - K @ self.H) @ P_pred
        return x_new, P_new, S
