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
from lib.math import fit
import copy
import mclient
from measurement import Measurement1D
from pulseseq.sequencer import *
from pulseseq.pulselib import *
import lmfit
import math

from lib.math import *


def analysis(xs, ys, fig, proj_num=0, vary_ofs=True, fit_type='poisson'):
    if fit_type=='poisson':
        fitstyle='displacement_cal'

    f = fitter.Fitter(fitstyle)
    p = f.get_lmfit_parameters(xs, ys)
    if fit_type=='poisson':
        p['n'].value = proj_num
        p['n'].vary = False
        p['ofs'].value = 0.0
        p['ofs'].vary = vary_ofs
        result = f.perform_lmfit(xs, ys, p=p, plot=False)
        p = result.params
        ys_fit = f.eval_func()

        txt = 'one photon disp amplitude: %0.3f' % p['dispscale'].value
    else:
        result = f.perform_lmfit(xs, ys, p=p, plot=False)
        p = result.params
        ys_fit = f.eval_func()

        pi_amp = 1.0 / (2.0 * p['f'].value)
        txt = 'Amp = %0.3f +/- %0.4f\n' % (p['A'].value, p['A'].stderr)
        txt += 'f = %0.4f +/- %0.5f\n' % (p['f'].value, p['A'].stderr)
        txt += 'phase = %0.3f +/- %0.4f\n' % (p['dphi'].value, p['dphi'].stderr)
        txt += 'period = %0.4f\n' % (1.0 / p['f'].value)
        txt += 'pi amp = %0.4f; pi/2 amp = %0.4f' % (pi_amp, pi_amp/2.0)
    fig.axes[0].plot(xs, ys_fit, 'g-', label=txt)
    fig.axes[0].legend()

    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[1].set_ylabel('Intensity [AU]')
    fig.axes[1].set_xlabel('Displacement [alpha]')

    fig.axes[1].plot(xs, ys_fit-ys, 'ks-')
    fig.canvas.draw()
    return p

class CavDisp(Measurement1D):

    def __init__(self, qubit_info, cav_info, disps, proj_num, seq=None, delay=0, bgcor=False, update=False, fit_type='poisson', **kwargs):
        self.qubit_info = qubit_info
        self.cav_info = cav_info
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.proj_num = proj_num
        self.delay = delay
        self.bgcor = bgcor

        self.update_ins = update
        self.fit_type = fit_type
        self.displacements = disps
        if len(disps) == 1:
            self.displacements = np.array([dmax])
        self.xs = np.abs(self.displacements) # we're plotting amplitude

        npoints = len(self.displacements)
        if self.bgcor:
            npoints *= 2

        super(CavDisp, self).__init__(npoints, infos=(qubit_info, cav_info), bgcor=bgcor, **kwargs)
        self.data.create_dataset('displacements', data=self.displacements)#, dtype=np.complex)
        self.data.set_attrs(
            delay=delay,
            bgcor=bgcor
        )

    def generate(self):
        '''
        If bg = True generate background measurement, i.e. no qubit pi pulse
        '''

        s = Sequence()

        r = self.qubit_info.rotate_selective
        c = self.cav_info.rotate
        for i, alpha in enumerate(self.displacements):
            for bg in (0, 1):
                if bg and not self.bgcor:
                    continue

                s.append(self.seq)
                # s.append(c(0, np.angle(alpha), amp=np.abs(alpha)))
                s.append(c.displace(np.abs(alpha), np.angle(alpha)))

                s.append(Delay(50))
                s.append(r(np.pi*(1-bg), X_AXIS))

                if self.delay:
                    s.append(Delay(self.delay))

                s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs


    def get_ys(self, data=None):
        ys = super(CavDisp, self).get_ys(data)
        if self.bgcor:
            return ys[::2] - ys[1::2]
        return ys


    def get_all_data(self, data=None):
        ys = super(CavDisp, self).get_ys(data)
        if self.bgcor:
            return ys[::2], ys[1::2]
        return ys[:], None

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

        if self.fit_type=='poisson':
            self.fit_params = analysis(self.xs, data, fig, proj_num=self.proj_num)
            ampScale = self.fit_params['dispscale'].value
            cav = mclient.instruments.get(self.cav_info.insname)
            oldAmp = self.cav_info.pi_amp
            new_amp = oldAmp/ampScale
        elif self.fit_type=='sine':
            self.fit_params = analysis(self.xs, data, fig, fit_type=self.fit_type)
            new_amp = 1.0/(2.0 * self.fit_params['f'].value)

        if self.update_ins:
            print 'displacement amp scaling: %.03f' % ampScale
            print '    updated amplitude is: %.03f' % new_amp
            cav.set_pi_amp(new_amp)
        else:
            print 'one photon displacement scale: %.03f (not setting)' % ampScale
            print '        updated amplitude is: %.03f' % new_amp

        return new_amp