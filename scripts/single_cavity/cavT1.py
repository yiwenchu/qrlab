import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from matplotlib import gridspec
from lib.math import fit
import copy
import mclient
from measurement import Measurement1D
from pulseseq.sequencer import *
from pulseseq.pulselib import *
import lmfit
import math

def poisson_decay_fit_func(params, xs, ys, n):
    '''
    Poissonian distribution given photon projection:
    alpha = alpha0 * exp(-xs / tau)

    params:
        ofs = offset
        amp = amplitude
        alpha0 = initial displacement
        tau = decay constant for energy, i.e. <n>
    '''

    alphas = params['alpha0'].value * np.exp(-xs / 2.0 / params['tau'].value)
#    nbars = params['alpha0'].value**2 * np.exp(-xs / params['tau'].value)
    nbars = alphas**2 + params['nth'].value
    vals = params['ofs'].value + params['amp'].value * nbars**n / math.factorial(n) * np.exp(-nbars)
    return ys - vals

def analysis(meas, data=None, fig=None):
    ys, fig = meas.get_ys_fig(data, fig)
    xs = meas.delays
#    fig.axes[1].plot(xs/1e3, ys, 'ks', ms=3)

    params = lmfit.Parameters()
    if meas.bgcor:
        params.add('ofs', value=0, vary=False)
    else:
        params.add('ofs', value=np.min(ys))
    params.add('amp', value=np.max(ys)-np.min(ys))
    params.add('alpha0', value=meas.disp)
    params.add('nth', value=0, min=0, max=0.4)
    params.add('tau', value=np.max(xs)/5.0, min=0)
    result = lmfit.minimize(poisson_decay_fit_func, params, args=(xs, ys, meas.proj_num))

    txt = 'Fit, Amp=%.03f\nT1=%.01f +- %.01f us\na0 = %.03f +- %.03f\nnth = %.03f +- %.03f' % (params['amp'].value, params['tau'].value/1e3, params['tau'].stderr/1e3, params['alpha0'].value, params['alpha0'].stderr, params['nth'].value, params['nth'].stderr)
    ys_model = -poisson_decay_fit_func(params, xs, 0, meas.proj_num)
    fig.axes[0].plot(xs/1e3, ys_model, label=txt)
    fig.axes[0].legend(loc=0)
    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[0].set_xlabel('Delay [us]')

    fig.axes[1].plot(xs/1e3, ys - ys_model)
    fig.canvas.draw()
    return params

class CavT1(Measurement1D):

    def __init__(self, qubit_info, cav_info, disp, delays,
                 proj_num, seq=None, extra_info=None, bgcor=True, postseq=None, **kwargs):
        self.qubit_info = qubit_info
        self.cav_info = cav_info
        self.disp = disp
        self.delays = delays
        self.proj_num = proj_num
        self.bgcor = bgcor
        self.postseq = postseq
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.xs = self.delays/1e3

        npoints = len(self.delays)
        if bgcor:
            npoints *= 2
        super(CavT1, self).__init__(npoints, infos=(qubit_info, cav_info), **kwargs)
        self.data.create_dataset('delays', data=self.delays)
        self.data.set_attrs(
            disp=disp,
        )

    def generate(self):
        s = Sequence()

        r = self.qubit_info.rotate_selective
        c = self.cav_info.rotate

        for i, delay in enumerate(self.delays):
            for bg in (0,1):
                if bg and not self.bgcor:
                    continue

                s.append(self.seq)
                s.append(c.displace(np.abs(self.disp), np.angle(self.disp)))
                s.append(Delay(delay))
                #
                s.append(r(np.pi*(1-bg), X_AXIS))
#                s.append(r(np.pi, 0))
                if self.postseq is not None:
                    s.append(self.postseq)
                s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def get_ys(self, data=None):
        ys = super(CavT1, self).get_ys(data)
        if self.bgcor:
            return ys[::2] - ys[1::2]
        return ys

    def get_all_data(self, data=None):
        ys = super(CavT1, self).get_ys(data)
        if self.bgcor:
            return ys[::2], ys[1::2]
        return ys[:], None
        

#    def create_figure(self):
#        '''
#        Overloads the Measurement create_figure() if <bgcor> is True
#
#        We will have three axes objects:
#        1. plots the raw data: data and background
#        2. plots the background subtracted data
#        3. plots residuals
#
#        If <residuals> is False, then plots 1 and 2 of equal size
#
#        This function assums that vert is True
#
#        '''
#        self.fig = plt.figure()
#        title = self.title
#        if self.data:
#            title += ' data in %s' % self.data.get_fullname()
#        self.fig.suptitle(title)
#
#        if not self.residuals:
#            self.fig.add_subplot(121)
#            self.fig.add_subplot(122)
#            return self.fig
#
#        gs = gridspec.GridSpec(3, 1, height_ratios=[3, 3, 1])
#        self.fig.add_subplot(gs[0])
#        self.fig.add_subplot(gs[1])
#        self.fig.add_subplot(gs[2])
#        return self.fig

    def update(self, avg_data):
        data, bg_data = self.get_all_data(avg_data)
        fig = self.get_figure()
        fig.axes[0].clear()
        fig.axes[1].clear()

        if hasattr(self, 'xs'):
            fig.axes[0].plot(self.xs, data, 'rs-', label='raw data')
            if self.bgcor:
                fig.axes[0].plot(self.xs, bg_data, 'bs-', label='background')
                fig.axes[0].plot(self.xs, data-bg_data, 'ks-', label='bg subtracted')
        else:
            fig.axes[0].plot(data, 'rs-', label='raw data')
            if self.bgcor:
                fig.axes[0].plot(bg_data, 'bs-', label='background')
                fig.axes[0].plot(data-bg_data, 'ks-', label='bg subtracted')

        fig.axes[0].legend(loc='best')
        fig.axes[1].legend(loc='best')

        fig.canvas.draw()

    def analyze(self, data=None, fig=None):
        self.fit_params = analysis(self, data, fig)
        return self.fit_params['tau'].value
