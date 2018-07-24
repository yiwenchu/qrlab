import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
import copy
import mclient
import lmfit
import pulseseq.sequencer_subroutines as ss
from lib.math import fitter
from matplotlib import gridspec

def analysis(phases, delays, data, fig=None):

    if fig is None:
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 2, width_ratios=[1,1])
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1])
        fig.add_subplot(gs[0])
        fig.add_subplot(gs[1])

    fig.axes[0].clear()
    fig.axes[1].clear()
    fig.canvas.draw()

    data = data.reshape(len(delays), 2, len(phases))
    params = []

    for i, d in enumerate(delays):
        for j, expt_type in enumerate([u'0-1', u'1-2']):
            xs = phases
            ys = data[i, j,:]

            f = fitter.Fitter('sine')
            p = f.get_lmfit_parameters(xs, ys)
            p['f'].value = 1.0
            p['f'].vary = False
            result = f.perform_lmfit(xs, ys, p=p, print_report=False, plot=False)
            p = result.params
            ys_fit = f.fit_func(xs, **f.get_params_dict(result.params))

            txt = '%s delay: %d ns; phase: %0.5f' % (expt_type, d, p['dphi'].value)
            fig.axes[0].plot(xs, ys, marker='s', ms=3)
            fig.axes[0].plot(xs, ys_fit, label=txt)

            params.append(p)

    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[0].set_xlabel('$\pi$ rotation angle (rad)')
    fig.axes[0].legend(loc=0)

    # fit phases
    phases = np.array([p['dphi'].value for p in params])
    print phases
#    phases = zip(*[iter(phases)]*2) # (delays, pi/nopi)
#    dphi = [ps[1] - ps[0] for ps in phases]
    dphi = phases[1::2] - phases[::2]
#    dphi = np.unwrap(dphi)
    print 'dphi: %s' % (dphi,)
#    f = fitter.Fitter('linear')
#    result = f.perform_lmfit(delays, dphi, print_report=False, plot=False)
#    ys_fit = f.fit_func(delays, **f.get_params_dict(result.params))
    print 'delays: %s' % (delays,)
    f = np.polyfit(delays, dphi, 1)
    ys_fit = np.poly1d(f)(delays)

    txt = 'slope = %0.3f KHz' % (f[0]/(2*np.pi)*1e6) # result.params['m'].value
    fig.axes[1].plot(delays, dphi, marker='s')
    fig.axes[1].plot(delays, ys_fit, label=txt)
    fig.axes[1].legend(loc=0)

    fig.canvas.draw()

    return params

class KerrRamseyPhoton(Measurement1D):

    def __init__(self, cav_info_01, cav_info_12, qubit_info,
                 phases, delays, seq=None, postseq=None, subtraction=False,
                 **kwargs):

        self.cav_info_01 = cav_info_01
        self.cav_info_12 = cav_info_12
        self.qubit_info = qubit_info

        self.phases = np.array(phases)
        self.delays = delays
        self.xs = np.array(phases) / (2 * np.pi)

        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq
        self.infos = [cav_info_01, cav_info_12, qubit_info,]
        self.subtraction = subtraction

        npoints = len(phases) * len(delays) * 2
        if subtraction:
            npoints *= 2

        super(KerrRamseyPhoton, self).__init__(npoints, infos=self.infos, **kwargs)
        self.data.create_dataset('delays', data=delays)
        self.data.create_dataset('phases', data=phases)

    def generate(self):
        s = Sequence()
        c01 = self.cav_info_01.rotate
        c12 = self.cav_info_12.rotate
        q = self.qubit_info.rotate

        subamps = [1.0]
        if self.subtraction:
            subamps = [0.0, 1.0]

        for delay in self.delays:
            for i, phase in enumerate(self.phases):
                for subamp in subamps:
                    # ramsey for 01
                    s.append(self.seq)

                    s.append(c01(np.pi/2, 0.0))
                    s.append(Delay(delay))
                    s.append(c01(np.pi/2, phase))

                    s.append(Delay(10))
                    s.append(q(subamp*np.pi, 0.0))

                    if self.postseq is not None:
                        s.append(self.postseq)
                    s.append(self.get_readout_pulse())

            for i, phase in enumerate(self.phases):
                for subamp in subamps:
                    # ramsey for 12
                    s.append(self.seq)

                    s.append(c01(np.pi, 0.0))
                    s.append(c12(np.pi/2, 0.0))
                    s.append(Delay(delay))
                    s.append(c12(np.pi/2, phase))
                    s.append(c01(np.pi, 0.0))

                    s.append(Delay(10))
                    s.append(q(subamp*np.pi, 0.0))

                    if self.postseq is not None:
                        s.append(self.postseq)
                    s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()

        return seqs

    def update(self, avg_data):
        ys = self.get_ys(avg_data)
        data = ys.reshape(len(self.delays), 2, len(self.phases))

        fig = self.get_figure()
        fig.axes[0].clear()
        if hasattr(self, 'xs'):
            for d in np.arange(len(self.delays)):
                for p in [0,1]:
                    fig.axes[0].plot(self.xs, data[d, p], marker='s', ms=3)
        else:
            for d in np.arange(len(self.delays)):
                for p in [0,1]:
                    fig.axes[0].plot(data[d, p])

        fig.canvas.draw()

    def analyze(self, data=None, fig=None):
        xs = self.xs
        delays = self.delays
        data = self.get_ys(data)
        fig = self.get_figure()
        self.fit_params = analysis(xs, delays, data, fig)

    def get_ys(self, data=None):
        ys = super(KerrRamseyPhoton, self).get_ys(data)
        if self.subtraction:
            return ys[::2] - ys[1::2]
        return self.complex_to_real(ys)

    def get_all_ys(self, data=None):
        ys = super(KerrRamseyPhoton, self).get_ys(data)
        if self.subtraction:
            return ys[0::2], ys[1::2]
        return self.complex_to_real(ys)
