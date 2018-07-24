# -*- coding: utf-8 -*-
"""
Cavity displacement and projection measurement, useful to calibrate
displacements.

by Brian Vlastakis, Reinier Heeres
"""

import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from matplotlib import gridspec
from lib.math import fitter
import copy
import mclient
from measurement import Measurement1D_bgcor
from pulseseq.sequencer import *
from pulseseq.pulselib import *

import fitting_programs as fp

def analysis(meas, data=None, fig=None, vary_ofs=False):
    ys, fig = meas.get_ys_fig(data, fig) # background subtracted.
    xs = meas.xs
#    fig.axes[1].plot(ys, 'ks-', label='bg subtracted')

    f = fitter.Fitter('poisson')
    p = f.get_lmfit_parameters(xs, ys)
    p['n'].value = meas.proj_num
    p['n'].vary = False
    p['ofs'].value = 0.0
    p['ofs'].vary = vary_ofs
    result = f.perform_lmfit(xs, ys, p=p, plot=False)
    params = result.params
    ys_fit = f.test_values(xs, p=params, noise_amp=0.0)

    txt = 'one photon disp amplitude: %0.3f' % params['xscale'].value
    fig.axes[1].plot(xs, ys_fit, 'g-', label=txt)
    fig.axes[1].legend()

    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[1].set_ylabel('Intensity [AU]')
    fig.axes[2].set_xlabel('Displacement [alpha]')

    fig.axes[2].plot(xs, ys_fit-ys, 'ks-')
    fig.canvas.draw()
    return params

class CavDispTime(Measurement1D_bgcor):

#    def __init__(self, qubit_info, cav_info, dmax, N, proj_num,
#                 seq=None, delay=0, bgcor=False, update=False, **kwargs):
    def __init__(self, qubit_info, cav_info, times, proj_num,
                 seq=None, delay=0, bgcor=False, **kwargs):
        self.qubit_info = qubit_info
        self.cav_info = cav_info
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.proj_num = proj_num
        self.delay = delay
        self.bgcor = bgcor

        self.times = times
        if len(times) == 1:
            self.times = np.array([dmax])
        self.xs = np.abs(self.times) # we're plotting amplitude

        npoints = len(self.times)
        if self.bgcor:
            npoints *= 2

        super(CavDispTime, self).__init__(npoints,
            infos=(qubit_info, cav_info), bgcor=bgcor, **kwargs)
        self.data.create_dataset('times', data=self.times,
                                 dtype=np.float64)
        self.data.set_attrs(
            delay=delay,
            bgcor=bgcor
        )

    def generate(self):
        '''
        If bg = True generate background measurement, i.e. no qubit pi pulse
        '''

        s = Sequence()

        r = self.qubit_info.rotate
        cpi = self.cav_info.rotate(np.pi,0)
        cchan = cpi.get_channels()[1]

        for i, time in enumerate(self.times):
            for i_bg in [1, 0]:
                if i_bg == 1 and not self.bgcor:
                    continue

                s.append(self.seq)
#                s.append(c(0, np.angle(alpha), amp=np.abs(alpha)))
                s.append(Constant(time,1,chan=cchan))
#                s.append(c(np.abs(alpha) * np.pi, np.angle(alpha)))

#                s.append(Delay(50))
                s.append(r(np.pi*i_bg, X_AXIS))

                if self.delay:
                    s.append(Delay(self.delay))

                s.append(self.get_readout_pulse())
#                s.append(Combined([
#                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.readout_chan),
#                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.acq_chan),
#                ]))

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def analyze(self, data=None, fig=None):
        self.fit_params = analysis(self, data, fig)
        new_amp = self.fit_params['xscale'].value
        cav = mclient.instruments.get(self.cav_info.insname)

        print 'one photon displacement amp: %.03f (not setting)' % new_amp

        return new_amp



#
##==============================================================================
##
##==============================================================================
#
#
#def plot_cal_disp(data, bg_data, disps, proj_num=0, fig=None):
#
#    if fig is None:
#        fig = plt.figure()
#    plt.suptitle('Cavity displacements')
#
#    ax = fig.add_subplot(211)
#    ax.plot(disps, data, 'rs-', label='raw data')
#    ax.plot(disps, bg_data, 'bs-', label='background')
#    ax.set_title('raw and background')
#    ax.legend(loc='best')
#
#    ax = fig.add_subplot(212)
#    ax.plot(disps, data - bg_data, 'ks-', label='with subtraction')
#    ax.set_title('subtraction')
#
#    import fitting_programs as fp
#    f = fp.Poisson(x_data=disps, y_data=data-bg_data, fix_n=proj_num)
#    result, params, y_final = f.fit(plot=False)
#
#    txt = 'one photon disp amplitude: %0.3f' % params['alpha_scale'].value
#    ax.plot(disps, y_final, 'g-', label=txt)
#    ax.legend(loc='best')
#
#    ax.set_xlabel('displacement amplitude')
#
#
#
##==============================================================================
##
##==============================================================================
#
#
#
#class PoissonFit(fit.Function):
#    '''
#    Poissonian distribution given photon projection: a * exp(- (b * x)**2 /2) * (b *x)**n / sqrt(factorial(m))
#
#     parameters:
#        a = amplitude
#        b = alpha scaling
#        n = photon projection (fixed)
#    '''
#
#    def __init__(self, proj_num, *args, **kwargs):
#        self.proj_num = proj_num
#        super(PoissonFit, self).__init__(*args, **kwargs)
#
#    def func(self, p, x=None):
#        p, x = self.get_px(p, x)
#        ret =  p[0] * np.exp(-np.square(p[1] * x)) * np.power(np.abs(p[1]*x), (2*self.proj_num)) / factorial(self.proj_num)
#        return ret
#
#def analysis_reinier(meas, data=None, fig=None):
#    ys, fig = meas.get_ys_fig(data, fig)
#    xs = meas.xs
#    fig.axes[0].plot(xs, ys, 'ks', ms=3)
#    amp0 = np.max(ys) - np.min(ys)
#    scaling0 = 1
#    print 'Amplitude estimate: %.03f ' % (amp0)
#
#    fPoiss = PoissonFit(meas.proj_num, xs, ys)
#    p0 = [amp0, scaling0]
##    plt.plot(xs/1e3, ft2.func(p0, xs), label='Guess')
#    p = fPoiss.fit(p0)
#    fig.axes[0].plot(xs, fPoiss.func(p, xs), label='Fit, Amp=%.03f , Scaling=%.03f '%(p[0], p[1]))
#    fig.axes[0].legend()
#    fig.axes[0].set_ylabel('Intensity [AU]')
#    fig.axes[0].set_xlabel('Displacement [alpha]')
#
#    fig.axes[1].plot(xs, ys - fPoiss.func(p, xs))
#    fig.canvas.draw()
#    return p[1]