import numpy as np


"""Measurement fusion utilities.

This module contains a simple information-form fusion routine used by the
synthetic demo. The function `fuse_measurements` takes a list of 2D position
measurements and their corresponding 2x2 covariance matrices and returns a
single fused position and covariance.

The fusion is performed in the information form:
  - information matrix = R^{-1}
  - information vector = R^{-1} @ z
  - summed information matrix and vector are converted back to covariance and
    mean via inversion.

If no valid measurements are provided (all entries in `z_list` are `None`),
the function returns `(None, None)` to indicate no fused measurement is
available.
"""


def fuse_measurements(z_list, R_list):
    """Fuse multiple 2D measurements with known covariances.

    Args:
        z_list (list): list of measurements where each measurement is a length-2
            array-like (x, y) or `None` if the sensor provided no detection.
        R_list (list): list of 2x2 covariance matrices corresponding to each
            measurement in `z_list`.

    Returns:
        (z_fused, R_fused) where `z_fused` is a length-2 numpy array containing
        the fused position, and `R_fused` is the 2x2 fused covariance matrix.
        Returns `(None, None)` if no measurements are present.
    """

    # Collect only the valid measurements and their inverse covariances
    zs = []
    Rs_inv = []
    for z, R in zip(z_list, R_list):
        if z is None:
            # Skip missing measurements
            continue
        zs.append(np.asarray(z))
        # Convert covariance to information matrix (inverse covariance)
        Rs_inv.append(np.linalg.inv(R))

    # If no valid measurements were provided, signal that by returning Nones
    if not zs:
        return None, None

    # Sum information matrices and information vectors
    R_inv_sum = np.zeros((2, 2))
    z_info = np.zeros(2)
    for z, Rinv in zip(zs, Rs_inv):
        R_inv_sum += Rinv
        z_info += Rinv @ z

    # Convert summed information back to covariance (R_fused) and mean (z_fused)
    R_fused = np.linalg.inv(R_inv_sum)
    z_fused = R_fused @ z_info

    return z_fused, R_fused
