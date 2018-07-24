import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement2D
from lib.math import fit
from matplotlib import colors as mcolors

def analysis(displacements, data, ax=None, plotdx=None):
    data = np.abs(data[::2]) - np.abs(data[1::2])
    XS = np.real(displacements)
    YS = np.imag(displacements)
    ZS = data.reshape(XS.shape)

    bwr = [(0.0, 0.0, 0.7), (0.1, 0.1, 1.0), (1.0, 1.0, 1.0), (1.0, 0.1, 0.1), (0.7, 0.0, 0.0)]
    cmap = mcolors.LinearSegmentedColormap.from_list('mymap', bwr, gamma=1)
    kwargs = dict(cmap=cmap, vmin=-ZS.max(), vmax=ZS.max())
    if plotdx is not None:
        plt.pcolor(plotdx, plotdx, ZS, **kwargs)
    else:
        plt.pcolor(XS, YS, ZS, **kwargs)
    ax = plt.gca()

    plt.colorbar()

    ax.set_xlim(np.min(XS), np.max(XS))
    ax.set_ylim(np.min(YS), np.max(YS))
    ax.set_xlabel(r'$Re \{\alpha \}$')
    ax.set_ylabel(r'$Im \{\alpha \}$')

class WignerFunction(Measurement2D):

    def __init__(self, qubit_info, cav_info, dmax, N, seq=None, delay=0, saveas=None, detunings=None, pi_areas=None, **kwargs):
        self.qubit_info = qubit_info
        self.cav_info = cav_info
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.delay = delay
        self.saveas = saveas
        self.detunings = detunings
        self.sigma = qubit_info.sigma
        if pi_areas is None:
            pi_areas = [qubit_info.pi_area for df in detunings]
        self.pi_areas = pi_areas

        dx = np.linspace(-dmax, dmax, N)
        delta = (dx[1] - dx[0]) / 2
        self.plotdx = np.linspace(-dmax-delta, dmax+delta, N+1)
        dy = 1j * dx
        XS, YS = np.meshgrid(dx, dy)
        self.displacements = -(XS + YS)

        npoints = 2 * len(dx)**2
        super(WignerFunction, self).__init__(npoints, infos=(qubit_info, cav_info), **kwargs)
        self.data.create_dataset('displacements', data=self.displacements, dtype=np.complex)
        self.data.set_attrs(delay=delay)

    def generate(self):
        '''
        If bg = True generate background measurement, i.e. without qubit pi pulse
        '''

        s = Sequence()

        geven = DetunedGaussians(self.sigma, chans=self.qubit_info.channels)
        for df, area in zip(self.detunings[::2], self.pi_areas[::2]):
            if abs(df) > 1e3:
                period = 1e9 / abs(df)
            else:
                period = 1e50
            print 'Detuning %s, Period: %.03f' % (df, period)
            geven.add_gaussian(0, period, area=area)
        godd = DetunedGaussians(self.sigma, chans=self.qubit_info.channels)
        for df, area in zip(self.detunings[1::2], self.pi_areas[1::2]):
            if abs(df) > 1e3:
                period = 1e9 / abs(df)
            else:
                period = 1e50
            print 'Detuning %s, Period: %.03f' % (df, period)
            godd.add_gaussian(0, period, area=area)

        c = self.cav_info.rotate
        for i, alpha in enumerate(self.displacements.flatten()):
            for i_bg in range(2):
                s.append(self.seq)

                add = c(np.abs(alpha), np.angle(alpha))
                if i_bg == 0:
                    add = Join([add, geven()])
                else:
                    add = Join([add, godd()])

                if self.delay:
                    add = Join([Delay(self.delay), add])

                add = Join([add, Combined([
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.readout_chan),
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.acq_chan),
                ])])
                s.append(add)

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def analyze(self, data=None, ax=None):
        ZS, pax = self.get_ys_ax(data, ax)
        ret = analysis(self.displacements, ZS, pax, plotdx=self.plotdx)
        if self.title:
            plt.title(self.title)
        if self.saveas:
            plt.savefig(self.saveas)
