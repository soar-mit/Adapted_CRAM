import numpy as np
import scipy.sparse as sp

from poles_residues import *

def Adapted_CRAM(q_mu: sp.csc_array, c: sp.csc_array, corr: np.ndarray, s: np.ndarray, n0: np.ndarray):
    """
    q_mu: K X K
    c: K X K*N_x 
    corr: K*N_x X K*N_x
    s: K*N_x
    """
    K = q_mu.shape[0]
    KN_x = s.size

    iden = sp.eye(q_mu.shape[0], format = "csc")

    np0 = n0.copy()
    us = np.zeros((K, KN_x))
    dn2 = np.zeros(K)
    for alpha, theta in zip(c48_alpha, c48_theta):
        luobj = sp.linalg.splu(q_mu - theta*iden)
        y0 = luobj.solve(np0)
        np0 += 2 * np.real(alpha * y0)
        dn2_shift = np.zeros(K, dtype = np.complex128) #accumulator for dn2 shift
        for kn_x in range(KN_x):
            col = s[kn_x]*c[:,[kn_x]]
            iso_index = kn_x % K
            yn = luobj.solve(us[:, kn_x] - s[kn_x]*c[:,kn_x].toarray().ravel()*y0[iso_index])
            us[:, kn_x] += 2 * np.real(alpha * yn)

            for kn_x2 in range(KN_x):
                dn2_shift -= corr[kn_x, kn_x2]*s[kn_x2]*c[:, kn_x2].toarray().ravel()*yn[kn_x2 % K]
        dn2 += 2 * np.real(alpha * luobj.solve(dn2 + dn2_shift))
    np0 *= c48_alpha0
    us *= c48_alpha0
    dn2 *= c48_alpha0
    return np0 + dn2
    
