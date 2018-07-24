import numpy as np
import matplotlib.pyplot as plt
from measurement import Measurement2D
from lib.math import fit
from pulseseq.sequencer import *
from pulseseq.pulselib import *

def analysis2d(displacements, data, ax=None, bgcor=False, plotdx=None, plotdy=None):
    if bgcor:
        data = data[::2] - data[1::2]
    XS = np.real(displacements)
    YS = np.imag(displacements)
    ZS = data.reshape(XS.shape)

    if plotdx is not None:
        plt.pcolormesh(plotdx, plotdy, ZS)
    else:
        plt.pcolormesh(XS, YS, ZS)
    ax = plt.gca()
    plt.colorbar()

    ax.set_xlim(np.min(XS), np.max(XS))
    ax.set_ylim(np.min(YS), np.max(YS))
    ax.set_xlabel(r'$Re \{\alpha \}$')
    ax.set_ylabel(r'$Im \{\alpha \}$')

def analysis1d(displacements, data, ax=None, bgcor=False):
    if bgcor:
        data = data[::2] - data[1::2]
    red = np.real(displacements)
    imd = np.imag(displacements)
    if (np.max(red) - np.min(red)) > (np.max(imd) - np.min(imd)):
        plotdx = red
        label = r'$Re{\alpha}$'
    else:
        plotdx = imd
        label = r'$Im{\alpha}$'

    ax = plt.gca()
    ax.plot(plotdx, data)
    ax.set_xlabel(label)
    ax.set_ylabel('Signal (AU)')

class QFunctionTomo(Measurement2D):

    def __init__(self, qubit_info, cav_info,
                 amax=None, N=None, amaxx=None, Nx=None, amaxy=None, Ny=None,
                 displacements=None, tomo_type='Z', plotdims=2,
                 seq=None, delay=0, saveas=None, bgcor=True, extra_info=None, **kwargs):
        self.qubit_info = qubit_info
        self.cav_info = cav_info
        self.seq = seq
        self.delay = delay
        self.saveas = saveas
        self.bgcor = bgcor
        self.extra_info = extra_info
        self.tomo_type = tomo_type
        self.plotdims = plotdims

        # Setup tomography pulses
        tomo_type = tomo_type.upper()
        if tomo_type == 'Z':
            tomo_pulse1 = qubit_info.rotate(np.pi, 0)
            tomo_pulse2 = Combined([
                            Constant(1, 0, chan=qubit_info.sideband_channels[0]),
                            Constant(1, 0, chan=qubit_info.sideband_channels[1])
                            ])
        elif tomo_type == 'X':
            tomo_pulse1 = qubit_info.rotate(np.pi/2, 0)
            tomo_pulse2 = qubit_info.rotate(-np.pi/2, 0)
        elif tomo_type == 'Y':
            tomo_pulse1 = qubit_info.rotate(np.pi/2, np.pi/2)
            tomo_pulse2 = qubit_info.rotate(-np.pi/2, np.pi/2)
        else:
            raise Exception('tomo_type should be X, Y or Z')

        self.tomo_pulse1 = tomo_pulse1
        self.tomo_pulse2 = tomo_pulse2

        # Setup displacements
        if amaxx is not None or amax is not None:
            if amaxx is not None:
                xs = np.linspace(-amaxx, amaxx, Nx)
                ys = np.linspace(-amaxy, amaxy, Ny)
            else:
                Nx, Ny = N, N
                xs = np.linspace(-dmax, dmax, N)
                ys = xs

            delta = (xs[1] - xs[0]) / 2
            self.plotdx = np.linspace(xs.min()-delta, xs.max()+delta, Nx+1)
            delta = (ys[1] - ys[0]) / 2
            self.plotdy = np.linspace(ys.min()-delta, ys.max()+delta, Ny+1)

            XS, YS = np.meshgrid(xs, ys)
            displacements = -(XS + 1j*YS)

        elif displacements is not None:
            displacements = displacements.astype(np.complex)
            self.plotdx = None
            self.plotdy = None

        else:
            raise Exception('Either amax/N, amaxx/amaxy/Nx/Ny or displacements should be specified')

        self.displacements = displacements
        npoints = displacements.size
        if bgcor:
            npoints *= 2
        super(QFunctionTomo, self).__init__(npoints, **kwargs)
        self.data.create_dataset('displacements', data=self.displacements, dtype=np.complex)
        self.data.set_attrs(
            delay=delay,
            bgcor=bgcor,
            tomo_type=tomo_type,
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
                    add = Join([add, self.tomo_pulse1])
                else:
                    add = Join([add, self.tomo_pulse2])

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
        if self.plotdims == 2:
            ret = analysis2d(self.displacements, ZS, pax, bgcor=self.bgcor, plotdx=self.plotdx, plotdy=self.plotdy)
        elif self.plotdims == 1:
            ret = analysis1d(self.displacements, ZS, pax, bgcor=self.bgcor)
        if self.saveas:
            plt.savefig(self.saveas)
