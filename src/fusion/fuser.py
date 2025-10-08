import numpy as np
def fuse_measurements(z_list, R_list):
    zs = []; Rs_inv = []
    for z, R in zip(z_list, R_list):
        if z is None: continue
        zs.append(z); Rs_inv.append(np.linalg.inv(R))
    if not zs: return None, None
    R_inv_sum = np.zeros((2,2)); z_info = np.zeros(2)
    for z, Rinv in zip(zs, Rs_inv):
        R_inv_sum += Rinv; z_info += Rinv @ z
    R_fused = np.linalg.inv(R_inv_sum); z_fused = R_fused @ z_info
    return z_fused, R_fused
