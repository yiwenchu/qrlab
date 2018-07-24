import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
from lib.math import fit
from matplotlib import colors as mcolors

def analysis(meas, data=None, fig=None):
    ys, fig = meas.get_ys_fig(data, fig)
    xs = meas.areas
    fig.axes[0].plot(xs, data, 'ks', ms=3)
    fig.axes[0].set_xlabel('Area')
    fig.axes[0].set_ylabel('Amplitude [AU]')

class PNRPhases_driven(Measurement1D):

    def __init__(self, qubit_info, cav_info, drive_info, alpha0, dalpha, areas=None, delay=None, saveas=None, seq=None, **kwargs):
        self.qubit_info = qubit_info
        self.cav_info = cav_info
        self.drive_info = drive_info
        self.alpha0 = alpha0
        self.dalpha = dalpha
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.delay = delay
        self.areas = areas
        self.xs = self.areas
        self.saveas = saveas

        kwargs['analysis_func'] = analysis
        super(PNRPhases_driven, self).__init__(len(areas), **kwargs)
        self.data.create_dataset('areas', data=self.areas)
        self.data.set_attrs(alpha0=alpha0, dalpha=dalpha, delay=delay)

    def generate(self):
        s = Sequence()

        c = self.cav_info.rotate
        q = self.qubit_info.rotate
        d = self.drive_info.rotate
        for area in self.areas:
            s.append(self.seq)

            drive = d(area, 0)
            if self.delay is None:
                delay = 0
            else:
                delay = (self.delay - drive.get_length())/ 2
            add = c(self.alpha0 + self.dalpha, 0)
            add = Join([add, Delay(delay/2)])
            add = Join([add, drive])
            add = Join([add, Delay(delay/2)])
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
        if self.drive_info.ssb:
            self.drive_info.ssb.modulate(seqs)
#        s.plot_seqs(seqs)

        self.seqs = seqs
        return seqs
