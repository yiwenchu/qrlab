# Example for different "window function" pulses.
# - Render several different pulses as well as their spectra
# Reinier Heeres, 2014

import numpy as np
from pulseseq.sequencer import *
from pulseseq.pulselib import *
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 8
N = 16
cmap = mpl.cm.get_cmap(name='spectral')
colors = [cmap(i) for i in np.linspace(0, 1.0, N)]
mpl.rcParams['axes.color_cycle'] = colors
mpl.rcParams['lines.linewidth'] = 1.5

NORM = True         # Whether to normalize power spectrum
W = 200             # Pulse width
SLEPSCALE = 4
ZEROS = 1000        # Number of zeros to append/prepend for FFT
TABLE = [
    ('Gaussian', Gaussian(W / 4.0, 0.5)),
    ('Square', Constant(W, 0.5)),
    ('Hanning', Hanning(W, 0.5)),
    ('Blackman', Blackman(W, 0.5)),
    ('BlackmanHarris', BlackmanHarris(W, 0.5)),
    ('Kaiser1.5', Kaiser(W, 0.5, alpha=1.5)),
    ('Kaiser2', Kaiser(W, 0.5, alpha=2)),
    ('Kaiser3', Kaiser(W, 0.5, alpha=3)),
    ('Slepian0.15', Slepian(W, 0.5, bw=0.15/SLEPSCALE)),
    ('Slepian0.2', Slepian(W, 0.5, bw=0.2/SLEPSCALE)),
    ('Slepian0.3', Slepian(W, 0.5, bw=0.3/SLEPSCALE)),
    ('Parabola2', Parabola(W, 0.5, power=2)),
    ('Parabola3', Parabola(W, 0.5, power=3)),
    ('Parabola4', Parabola(W, 0.5, power=4)),
    ('FlatTop', FlatTop(W, 0.5)),
]

seq = Sequence()
seq.append(Delay(ZEROS))
c = Combined([])
for ch, (name, pulse) in enumerate(TABLE):
    pulse.chan = ch
    c.add_item(pulse)
seq.append(c)
seq.append(Delay(ZEROS))

s = Sequencer(seq, minlen=1)
seqs = s.render()
s.plot_seqs(seqs)
s.print_seqs(seqs)

plt.figure()
plt.suptitle('Spectral power density')
for ch, (name, pulse) in enumerate(TABLE):
    ys = seqs[ch].get_data()
    F = np.abs(np.fft.fft(ys))**2
    F = F[:round(len(F)/4)]
    X = np.arange(len(F)) * 1.0 / len(ys) * 1e3
    if NORM:
        F /= np.max(F)
    plt.plot(X, F, label=name)

plt.xlim(0, 100)
if NORM:
    plt.ylim(1e-10, 1)
else:
    plt.ylim(1e-12, 1e3)
plt.yscale('log')
plt.legend()
plt.xlabel('Frequency [MHz]')

