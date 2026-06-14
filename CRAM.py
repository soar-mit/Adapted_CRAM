import numpy as np
import scipy.sparse as sp

from poles_residues import *

def CRAM(a: sp.csc_array, z0: np.ndarray):
    iden = sp.eye(a.shape[0], format = "csc")

    y = z0.copy()
    for alpha, theta in zip(c48_alpha, c48_theta):
        y += 2 * np.real(alpha * sp.linalg.spsolve(a - theta*iden, y))
    y *= c48_alpha0
    return y

if __name__ == "__main__":
    A = sp.csc_array((), np.float64)
    n0 = np.zeros(10)
    exp_CRAM(A, n0)

