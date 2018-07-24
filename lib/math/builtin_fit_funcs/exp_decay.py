import numpy as np
import common

def func(xs, A=1, tau=1, ofs=0):
    return A * np.exp(-xs / tau) + ofs

def guess(xs, ys):
    yofs = ys[-1]
    ys = ys - yofs
    return dict(
        A=ys[0],
        tau=xs[common.find_index_of(ys, ys[0]/2)],
        ofs=yofs,
    )

TEST_RANGE = 0, 100
TEST_PARAMS = dict(A=-5, tau=10)
