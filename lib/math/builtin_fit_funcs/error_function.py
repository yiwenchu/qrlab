import numpy as np
import scipy.special as special
import matplotlib.pyplot as plt
import common

# The fitting function, should have "xs" as first parameter.
# Each further parameter is interpreted as a fitting parameters. A default
# value should be specified if no "guess" function is provided
def func(xs, A=1, b=10, x0=10, ofs=0):
    return A * special.erf(b * (xs - x0)) + ofs

# The guess function should produce a dictionary of parameter values that are
# a decent starting point for fitting.
def guess(xs, ys):
    yofs = np.average(ys)
    return dict(
        A = np.max(ys) - np.min(ys),
        b = 10,
        x0 = guess_x0(xs, ys),
        ofs = yofs,
    )

def guess_x0(xs, ys):
	derivs = np.array([ys[i+1] - y[i] for i in np.arange(len(ys))])
	return xs[np.argmax(np.abs(derivs))]