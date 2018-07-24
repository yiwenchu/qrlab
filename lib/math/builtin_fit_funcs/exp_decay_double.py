import numpy as np
import common

def func(xs, A=1, B=1, tau=1, tau1=1, ofs=0):
    return A * np.exp(-xs / tau) + B * np.exp(-xs / tau1) + ofs

def guess(xs, ys):
    yofs = ys[-1]
    ys_ofs = ys - yofs
    return dict(
        A 		= ys_ofs[0]/2,
        B 		= ys_ofs[0]/2,
        tau 	= xs[common.find_index_of(ys_ofs, ys_ofs[0]/2)]/2,
        tau1	= xs[common.find_index_of(ys_ofs, ys_ofs[0]/2)],
        ofs 	= yofs,
    )
