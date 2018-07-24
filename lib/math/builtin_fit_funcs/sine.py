import numpy as np
import matplotlib.pyplot as plt

# The fitting function, should have "xs" as first parameter.
# Each further parameter is interpreted as a fitting parameters. A default
# value should be specified if no "guess" function is provided
def func(xs, A=1, f=0.05, dphi=np.pi/4, ofs=0):
    return A * np.sin(2*np.pi*xs*f + dphi) + ofs

# The guess function should produce a dictionary of parameter values that are
# a decent starting point for fitting.
def guess(xs, ys):
    yofs = np.average(ys)
    ys = ys - yofs
    P = np.fft.fft(ys)
    P = P[:round(len(ys)/2.0)]
    F = np.fft.fftfreq(len(ys), d=xs[1]-xs[0])
    F = F[:round(len(ys)/2.0)]
    imax = np.argmax(np.abs(P))
    return dict(
        A = 2 * np.abs(P[imax]) / len(xs),
        f = F[imax],
        dphi = np.angle(P[imax])+np.pi/2,
        ofs = yofs,
    )

# These parameters can be specified to set default testing values
# If nothing is specified, TEST_RANGE will be 0, 1 and TEST_PARAMS
# will be set to the default values of the function.
TEST_RANGE = 0, 100
TEST_PARAMS = dict(A=10, dphi=np.pi/4, noise_amp=2.0)

