import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement2D
from lib.math import fit

def analysis(displacements, data, ax=None, bgcor=False, plotdx=None):
    if bgcor:
        data = data[::2] - data[1::2]
    XS = np.real(displacements)
    YS = np.imag(displacements)
    #ZS = np.abs(data.reshape(XS.shape))
    ZS = data.reshape(XS.shape)

    if plotdx is not None:
        plt.pcolormesh(plotdx, plotdx, ZS)
    else:
        plt.pcolormesh(XS, YS, ZS)
    ax = plt.gca()
    plt.colorbar()

    ax.set_xlim(np.min(XS), np.max(XS))
    ax.set_ylim(np.min(YS), np.max(YS))
    ax.set_xlabel(r'$Re \{\alpha \}$')
    ax.set_ylabel(r'$Im \{\alpha \}$')

class CorrQFunction(Measurement2D):

    def __init__(self, qubit_info, cav_info, dmax, N, qubit_rot = np.pi,seq=None, delay=0, saveas=None, bgcor=False, extra_info=None, **kwargs):
        self.qubit_info = qubit_info
        self.cav_info = cav_info
        self.seq = seq
        self.delay = delay
        self.saveas = saveas
        self.bgcor = bgcor
        self.extra_info = extra_info
        self.qubit_rot = np.abs(qubit_rot)
        self.qubit_angle = np.angle(qubit_rot)

        xs = np.linspace(-dmax, dmax, N)
        delta = (xs[1] - xs[0]) / 2
        self.plotdx = np.linspace(-dmax-delta, dmax+delta, N+1)
        ys = 1j * xs
        XS, YS = np.meshgrid(xs, ys)
        self.displacements = -(XS + YS)

        npoints = len(xs)**2
        if bgcor:
            npoints *= 2
        super(CorrQFunction, self).__init__(npoints, **kwargs)
        self.data.create_dataset('displacements', data=self.displacements, dtype=np.complex)
        self.data.set_attrs(
            delay=delay,
            bgcor=bgcor
        )

    def generate(self):
        '''
        If bg = True generate background measurement, i.e. without qubit pi pulse
        '''

        s = Sequence()

        r = self.qubit_info.rotate
        c = self.cav_info.rotate
        for i, alpha in enumerate(self.displacements.flatten()):
            for i_bg in range(2):
                if i_bg == 1 and not self.bgcor:
                    continue

                if self.seq is not None:
                    s.append(self.seq)
                else:
                    s.append(Trigger(250))

                add = c(np.abs(alpha), np.angle(alpha))
                if i_bg == 0:
                    add = Join([add, r(self.qubit_rot, self.qubit_angle)])
                else:
                    add = Join([add, r(self.qubit_rot - np.pi, self.qubit_angle)])

                if self.delay:
                    add = Join([Delay(self.delay), add])

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
        ZS, pax = self.get_ys_ax(data, ax)
        ret = analysis(self.displacements, ZS, pax, bgcor=self.bgcor, plotdx=self.plotdx)
        if self.title:
            plt.title(self.title)
        if self.saveas:
            plt.savefig(self.saveas)
