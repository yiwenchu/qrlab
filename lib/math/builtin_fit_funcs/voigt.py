import numpy as np
import common
from scipy.special import wofz as fad

def func(xs, x0=10, sigma=0, gamma=0, amp=1, ofs=0):
    '''
    Voigt profile (convolution of a lorentzian response and a gaussian
    spectroscopic impulse), calculated from the Faddeeva function.

    Jacob 9/2014

    FWHM calculation included.

    '''

    fadarg = ((xs-x0) + 1j*gamma)/(sigma*np.sqrt(2))

    ys = np.real(fad(fadarg)/(sigma*np.sqrt(2*np.pi)))

    return amp*ys/np.max(ys) + ofs

def fwhm(pdict):
    sigma = pdict['sigma']
    gamma = pdict['gamma']

    fg = 2*sigma*np.sqrt(2*np.log(2))
    fl = 2*gamma

    fv = 0.5346 * fl + np.sqrt(0.2166 * fl**2 + fg**2)

    return fg, fl, fv

def hwhm(pdict):
    return np.array(fwhm(pdict)) / 2.0

def guess(xs, ys):
    yofs = common.determine_offset(ys)
    ys = ys - yofs
    maxidx = np.argmax(np.abs(ys))
    return dict(
        ofs=yofs,
        amp = ys[maxidx],
        x0=xs[maxidx],
        sigma=common.determine_peak_width(xs, ys, maxidx)/2,
        gamma=common.determine_peak_width(xs, ys, maxidx),
    )


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    xs = np.arange(50)

    ys = func(xs, x0=35, sigma=4, gamma=4, amp=3, ofs=1)

    plt.figure()
    plt.plot(xs,ys)
