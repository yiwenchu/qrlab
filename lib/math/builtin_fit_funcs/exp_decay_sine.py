import numpy as np
import matplotlib.pyplot as plt
import sine

def func(xs, A=1, f=0.05, dphi=np.pi/4, ofs=0, tau=0.5):
    return A * np.sin(2*np.pi*xs*f + dphi) * np.exp(-xs / tau) + ofs

def guess(xs, ys):
    d = sine.guess(xs, ys)
    d['tau'] = np.average(xs)
    return d
