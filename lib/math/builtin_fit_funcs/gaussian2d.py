import numpy as np
import common

def func(xs, ys, x0=0, y0=0, sigmax=1, sigmay=1, area=1, ofs=0):
    r2 = (xs - x0)**2 / (2 * sigmax**2) + (ys - y0)**2 / (2 * sigmay)**2
    return ofs + area / (2 * sigmax * sigmay) / np.sqrt(np.pi / 2) * np.exp(-r2)

def guess(xs, ys, zs):
    zofs = common.determine_offset(zs)
    zs = zs - zofs
    maxidxy = np.argmax(np.abs(zs).sum(axis=1))
    maxidxx = np.argmax(np.abs(zs).sum(axis=0))

    sigmax = common.determine_peak_width(xs[maxidxy,:], zs[maxidxy,:], maxidxx, factor=np.exp(-1)) / 2.0
    sigmay = common.determine_peak_width(ys[:,maxidxx], zs[:,maxidxx], maxidxy, factor=np.exp(-1)) / 2.0

    return dict(
        x0=xs[maxidxy,maxidxx],
        y0=ys[maxidxy,maxidxx],
        ofs=zofs,
        area=common.determine_peak_area(xs[maxidxy,:], zs[maxidxy,:]) * \
                common.determine_peak_area(ys[:,maxidxx], zs[:,maxidxx]),
        sigmax=sigmax,
        sigmay=sigmay,
    )

TEST_RANGE = 0, 10
TEST_PARAMS = dict(x0=4.5, y0=5.5, sigmax=1.2, sigmay=1.8, area=20, ofs=10)
