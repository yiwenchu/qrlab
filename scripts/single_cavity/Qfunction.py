import numpy as np
import matplotlib.pyplot as plt
from measurement import Measurement2D
from lib.math import fit
from pulseseq.sequencer import *
from pulseseq.pulselib import *

def analysis(meas, data=None, fig=None):
    zs, fig = meas.get_ys_fig(data, fig)
    zs = zs.reshape(len(meas.xs), len(meas.ys))
    xs, ys = meas.get_plotxsys()
    ax = fig.axes[0]
    plt.sca(ax)
    pc = ax.pcolormesh(xs, ys, zs)
    fig.colorbar(pc)

    ax.set_xlim(xs.min()), xs.max()
    ax.set_ylim(ys.min(), ys.max())
    ax.set_xlabel(r'$Re \{\alpha \}$')
    ax.set_ylabel(r'$Im \{\alpha \}$')
    fig.canvas.draw()

class QFunction(Measurement2D):

    def __init__(self, qubit_info, cav_info,
                 amax=None, N=None, amaxx=None, Nx=None, amaxy=None, Ny=None,
                 seq=None, delay=0, saveas=None, bgcor=False, **kwargs):
        self.qubit_info = qubit_info
        self.cav_info = cav_info
        self.seq = seq
        self.delay = delay
        self.saveas = saveas
        self.bgcor = bgcor

        if amaxx is not None:
            xs = np.linspace(-amaxx, amaxx, Nx)
            ys = np.linspace(-amaxy, amaxy, Ny)
        else:
            Nx, Ny = N, N
            xs = np.linspace(-amax, amax, N)
            ys = xs

        XS, YS = np.meshgrid(xs, ys)
        self.displacements = -(XS + 1j*YS)
        self.xs = xs
        self.ys = ys

        npoints = self.displacements.size
        if bgcor:
            npoints *= 2
        super(QFunction, self).__init__(npoints, infos=(qubit_info, cav_info), residuals=False, **kwargs)
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

                s.append(c(np.abs(alpha), np.angle(alpha)))
                if i_bg == 0:
                    s.append(r(np.pi, X_AXIS))

                if self.delay:
                    s.append(Delay(self.delay))

                s.append(Combined([
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.readout_chan),
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.acq_chan),
                ]))

        s = self.get_sequencer(s)
#        import cProfile
#        cProfile.runctx('s.render()', dict(s=s), dict())
        seqs = s.render(debug=False)
        return seqs

    def get_ys(self, data=None):
        ys = super(QFunction, self).get_ys(data)
        if self.bgcor:
            return ys[::2] - ys[1::2]
        return ys

    def analyze(self, data=None, fig=None):
        self.fit_params = analysis(self, data, fig)