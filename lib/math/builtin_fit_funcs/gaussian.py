import numpy as np
import common

def func(xs, ofs=0, area=10, x0=0, sigma=2):
    '''
    Gaussian defined by it's area <area>, sigma <s>, position <x0> and
    y-offset <ofs>.
    '''
    return ofs + area / (2 * sigma) / np.sqrt(np.pi / 2) * np.exp(-2 * (xs - x0)**2 / (2 * sigma)**2)

def guess(xs, ys):
    yofs = common.determine_offset(ys)
    ys = ys - yofs
    maxidx = np.argmax(np.abs(ys))
    sigma = common.determine_peak_width(xs, ys, maxidx, factor=np.exp(-1)) / 2.0
    return dict(
        ofs=yofs,
        area=common.determine_peak_area(xs, ys),
        x0=xs[maxidx],
        sigma=sigma,
    )

def get_fwhm(**p):
    return p['sigma'] * 2 * np.sqrt(2*np.log(2))

TEST_RANGE = 0, 10
TEST_PARAMS = dict(x0=4.2, sigma=0.6, area=10, ofs=10)
