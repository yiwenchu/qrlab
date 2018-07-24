import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
from lib.math import fit
from matplotlib import colors as mcolors

def analysis(meas, data, fig):
    ys, fig = meas.get_ys_fig(data, fig)
    xs = meas.delays
    fig.axes[0].plot(xs, ys, 'ks', ms=3)
    fig.axes[0].set_xlabel('Delay [ns]')
    fig.axes[0].set_ylabel('Amplitude [AU]')
    fig.canvas.draw()

class PNRPhases(Measurement1D):

    def __init__(self, qubit_info, cav_info, alpha0, dalpha, delays=None, saveas=None, seq=None, **kwargs):
        self.qubit_info = qubit_info
        self.cav_info = cav_info
        self.alpha0 = alpha0
        self.dalpha = dalpha
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.delays = delays
        self.xs = self.delays
        self.saveas = saveas

        kwargs['residuals'] = False
        super(PNRPhases, self).__init__(len(delays), **kwargs)
        self.data.create_dataset('delays', data=self.delays)
        self.data.set_attrs(alpha0=alpha0)
        self.data.set_attrs(dalpha=dalpha)

    def generate(self):
        s = Sequence()

        c = self.cav_info.rotate
        q = self.qubit_info.rotate
        for delay in self.delays:
            s.append(self.seq)

            add = c(self.alpha0 + self.dalpha, 0)
            add = Join([add, Delay(delay)])
            add = Join([add, c(self.dalpha, np.pi)])
            add = Join([add, q(np.pi, 0)])

            add = Join([add, Combined([
                Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.readout_chan),
                Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.acq_chan),
            ])])
            s.append(add)

        s = Sequencer(s)
        seqs = s.render()
        if self.qubit_info.ssb:
            self.qubit_info.ssb.modulate(seqs)
        if self.cav_info.ssb:
            self.cav_info.ssb.modulate(seqs)
#        s.plot_seqs(seqs)

        self.seqs = seqs
        return seqs

    def analyze(self, data=None, fig=None):
        ret = analysis(self, data, fig)
        if self.saveas:
            plt.savefig(self.saveas)
