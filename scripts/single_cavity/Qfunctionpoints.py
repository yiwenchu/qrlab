import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
from lib.math import fit

import matplotlib as mpl
mpl.rcParams['legend.fontsize'] = 6
cmap = mpl.cm.get_cmap(name='hsv')

def analysis(displacements, delays, bgcor, data, ax=None):
    colorlen = len(displacements)
#    if (colorlen % 2) == 0:
#        colorlen /= 2
    colors = [cmap(i) for i in np.linspace(0, 1.0, colorlen)]

    if bgcor:
        data = data[::2] - data[1::2]

    if ax is None:
        ax = plt.figure().add_subplot(111)

    for i_disp, disp in enumerate(displacements):
        ys = data[i_disp*len(delays):(i_disp+1)*len(delays)]
        ax.plot(delays, ys, label='|a| = %.03f, phi = %.02f x pi' % (np.abs(disp), np.angle(-disp)/np.pi), color=colors[i_disp%colorlen])

    ax.set_xlabel(r'Delay (ns)')
    ax.set_ylabel(r'Amplitude (AU)$')
    ax.legend(loc=0)

class QFunctionPoints(Measurement1D):

    def __init__(self, qubit_info, cav_info, alpha, N=1, delays=(0,), phi_max=2*np.pi, seq=None, delay=0, saveas=None, bgcor=False, extra_info=None, **kwargs):
        self.qubit_info = qubit_info
        self.cav_info = cav_info
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.delays = delays
        self.N = N
        self.saveas = saveas
        self.bgcor = bgcor
        self.extra_info = extra_info

        rotations = np.exp(1j * np.linspace(0, phi_max, N, endpoint=False))
        self.displacements = -alpha * rotations
        npoints = len(self.displacements) * len(delays)
        if bgcor:
            npoints *= 2
        super(QFunctionPoints, self).__init__(npoints, **kwargs)
        self.data.create_dataset('displacements', data=self.displacements, dtype=np.complex)
        self.data.set_attrs(
            delays=delays,
            bgcor=bgcor
        )

    def generate(self):
        '''
        If bg = True generate background measurement, i.e. without qubit pi pulse
        '''

        s = Sequence()

        r = self.qubit_info.rotate
        c = self.cav_info.rotate
        for i_disp, alpha in enumerate(self.displacements.flatten()):
            for i_delay, delay in enumerate(self.delays):
                for i_bg in range(2):
                    if i_bg == 1 and not self.bgcor:
                        continue

                    s.append(self.seq)
                    add = c(np.abs(alpha), np.angle(alpha))
                    if i_bg == 0:
                        add = Join([add, r(np.pi, X_AXIS)])
                    else:
                        add = Join([add, Combined([
                            Constant(1, 0, chan=r.chans[0]),
                            Constant(1, 0, chan=r.chans[1])
                        ])])

                    if delay != 0:
                        add = Join([Delay(delay), add])

                    s.append(add)
                    s.append(Combined([
                        Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.readout_chan),
                        Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.acq_chan),
                    ]))

        s = Sequencer(s)
        seqs = s.render()
        if self.qubit_info.ssb:
            self.qubit_info.ssb.modulate(seqs)
        if self.cav_info.ssb:
            self.cav_info.ssb.modulate(seqs)
        if type(self.extra_info) in (types.TupleType, types.ListType):
            for info in self.extra_info:
                if info.ssb:
                    info.ssb.modulate(seqs)
        elif self.extra_info and self.extra_info.ssb:
            self.extra_info.ssb.modulate(seqs)
        #s.plot_seqs(seqs)

        self.seqs = seqs
        return seqs

    def analyze(self, data=None, ax=None):
        data, pax = self.get_ys_ax(data, ax)
        ret = analysis(self.displacements, self.delays, self.bgcor, np.abs(data), ax=pax)
        if self.title:
            plt.title(self.title)
        if self.saveas:
            plt.savefig(self.saveas)
