import numpy as np
import matplotlib.pyplot as plt
import sine

def func(xs, A1=1, A2=2, f1=0.05, f2=0.05, dphi1=np.pi/4, dphi2=np.pi/4, ofs=0, tau=0.5):
    return A1 * np.sin(2*np.pi*xs*f1 + dphi1) + A2 *  np.sin(2*np.pi*xs*f2 + dphi2) * np.exp(-xs / tau) + ofs

def guess(xs, ys):
    d = sine.guess(xs, ys)
    g={}
    g['f1'] = d['f']
    g['f2'] = 0.9*d['f']
    g['dphi1'] = d['dphi']
    g['dphi2'] = d['dphi']
    g['A1'] = d['A']
    g['A2'] = d['A']
    g['ofs'] = d['ofs']
    g['tau'] = np.average(xs)
    return g
