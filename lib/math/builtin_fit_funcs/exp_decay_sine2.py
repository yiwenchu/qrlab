import numpy as np
import matplotlib.pyplot as plt
import sine

def func(xs, A=1, f=0.05, dphi=np.pi/4, ofs=0, tau=0.5):
    return A * (1-np.sin(2*np.pi*xs*f + dphi)) * np.exp(-xs / tau) + ofs

def guess(xs, ys):
    # yofs = (np.amax(ys)+np.amin(ys))/2
    yofs = ys[-1]
    tau = np.average(xs)
    ys = ys - yofs
    ys = ys/np.exp(-xs / tau)
    ys = ys[:len(ys)/2]-np.average(ys[:len(ys)/2])

    P = np.fft.fft(ys)
    P = P[:round(len(ys)/2.0)]
    F = np.fft.fftfreq(len(ys), d=xs[1]-xs[0])
    F = F[:round(len(ys)/2.0)]
    imax = np.argmax(np.abs(P))
    d = dict(
        A = 2 * np.abs(P[imax]) / len(xs),
        f = F[imax],
        dphi = np.angle(P[imax])+np.pi/2,
        ofs = yofs,
        )
    d['tau'] = tau
    return d
