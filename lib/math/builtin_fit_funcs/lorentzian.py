import numpy as np
import common

def func(xs, ofs=0, area=10, x0=0, w=2):
    '''
    Lorentzian defined by it's area <area>, width <w>, position <x0> and
    y-offset <ofs>.
    '''
    return ofs + 2 * area * w / np.pi / (4 * (xs - x0)**2 + w**2)

def guess(xs, ys):
    yofs = common.determine_offset(ys)
    ys = ys - yofs
    maxidx = np.argmax(np.abs(ys))
    return dict(
        ofs=yofs,
        area=common.determine_peak_area(xs, ys),
        x0=xs[maxidx],
        w=common.determine_peak_width(xs, ys, maxidx),
    )

TEST_RANGE = 0, 10
TEST_PARAMS = dict(x0=4.5, w=1.2, area=-10)
