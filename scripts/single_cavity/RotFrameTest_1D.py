# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:42:10 2013

@author: Brian Vlastakis
"""

import numpy as np
import matplotlib.pyplot as plt
from measurement import Measurement2D
from lib.math import fit
from pulseseq.sequencer import *
from pulseseq.pulselib import *

class DetuningFit(fit.Function):
    '''
    Detuning fit looking at distance between coherent states:
        a * exp(- | 2*b* sin[ (c - d(t+t0))/2 ] |**2
            (distance between coherent states related by the chord of circle)

     parameters:
        a = amplitude
        b = displacement magnitude (fixed)
        c = displacement phase (fixed)
        d = drive detuning
        t = wait time (independent variable)
        t0= initial wait time
    '''
    def __init__(self, dispMag, dispAngle, *args, **kwargs):
        self.dispMag = dispMag
        self.dispAngle = dispAngle
        super(DetuningFit, self).__init__(*args, **kwargs)

    def func(self, p, x=None):
        p, x = self.get_px(p, x)
        ret = p[0] * np.exp( -1 * np.abs(np.square(2*self.dispMag*np.sin( (self.dispAngle-2*np.pi*p[1]*(x - p[2])) /2))) )
        return ret

class Detuning3DFit(fit.Fit3D):
    '''
    Detuning fit looking at distance between coherent states:
        a * exp(- | 2*b* sin[ (c - d(t+t0))/2 ] |**2
            (distance between coherent states related by the chord of circle)

     parameters:
        a = amplitude
        b = displacement magnitude (fixed)
        c = displacement phase (fixed)
        d = drive detuning
        t = wait time (independent variable)
        t0= initial wait time
    '''
    def __init__(self, dispMag, *args, **kwargs):
        self.dispMag = dispMag
        super(Detuning3DFit, self).__init__(*args, **kwargs)

    def func(self, p, x=None, y=None):
        p, x, y = self.get_pxy(p, x, y)
        ret = p[0] * np.exp( -1 * np.abs(np.square(2*self.dispMag*np.sin( (y-2*np.pi*p[1]*(x - p[2])) /2))) )
        return ret

def analysis(delays, displacements, data, ax=None, bgcor=False, plotdx=None, plotdy=None):
    if bgcor:
        data = data[::2] - data[1::2]

    XS = delays
    YS = np.angle(displacements)
    ZS = data.reshape(XS.shape)

    amp0 = np.max(ZS) - np.min(ZS)
    t0 = -20
    detuning0 = -0.01
    p0 = [-amp0, detuning0, t0]
    print p0
    dispMag = np.max(np.abs(displacements))
    print dispMag
    dispAng = plotdy
    print dispAng

    fDetuning3D = Detuning3DFit(dispMag, XS, YS, ZS)
    p3D = fDetuning3D.fit(p0)
    print p3D
    fig, axes = plt.subplots(nrows = 3, ncols = 1)
    for i, ax in enumerate(axes):
        ax.plot(plotdx, ZS[i])
        ax.set_ylim(np.min(ZS), np.max(ZS))
        ax.set_xlabel(r'Delay $\{ns \}$')
        ax.set_ylabel(r'Intensity [AU]')
        fDetuning = DetuningFit(dispMag, dispAng[i], plotdx, ZS[i])
#       plt.plot(xs/1e3, ft2.func(p0, xs), label='Guess')
        p = fDetuning.fit(p0)
        #print p
        ax.plot(plotdx, fDetuning.func(p3D, plotdx), label='Fit, Detuning=%.03f MHz, T0=%.03f ns '%(p3D[1]*1000, p3D[2]))
        ax.legend(loc=0)

#    ax.set_xlim(np.min(plotdx), np.max(plotdx))
#    ax.set_ylim(np.min(plotdy), np.max(plotdy))


class RotFrameTest(Measurement2D):

    def __init__(self, qubit_info, cav_info, tmax, Nt, mag, seq=None, bgcor=False, extra_info=None, saveas=None, **kwargs):
        self.qubit_info = qubit_info
        self.cav_info = cav_info
        self.seq = seq
        self.bgcor = bgcor
        self.mag = mag
        self.extra_info = extra_info
        self.saveas = saveas

        xs = np.linspace(0, tmax, Nt)
        #delta = (xs[1] - xs[0])/2
        self.plotdx = xs #np.linspace(0-delta, tmax+delta, Nt+1)

        angle_list = np.linspace(0, np.pi, 3)
        #delta2 = (angle_list[1] - angle_list[0])/2
        ys = mag * np.exp(1j * angle_list)
        self.plotdy = angle_list

        XS, YS = np.meshgrid(xs, ys)
        self.delay_times = XS
        self.displacements = YS

        npoints = len(xs)*len(ys)
        if bgcor:
            npoints *= 2
        super(RotFrameTest, self).__init__(npoints, **kwargs)
        self.data.create_dataset('delay_times', data=self.delay_times, dtype=np.float)
        self.data.create_dataset('displacements', data = self.displacements, dtype=np.complex)

    def generate(self):
        '''
        If bg = True generate background measurement, i.e. without qubit pi pulse
        '''

        s = Sequence()

        r = self.qubit_info.rotate
        c = self.cav_info.rotate
        delay_list = self.delay_times.flatten()
        alpha_list = self.displacements.flatten()
        npoints = len(delay_list)

        for delay, alpha in zip(delay_list, alpha_list):
            for i_bg in range(2):
                if i_bg == 1 and not self.bgcor:
                    continue
                if self.seq is not None:
                    s.append(self.seq)
                else:
                    s.append(Trigger(250))

                disp1 = c(np.abs(alpha), 0)
                disp2 = c(-np.abs(alpha), np.angle(alpha))

                combo = Join([disp1,Delay(delay),disp2])
                if i_bg ==0:
                    combo = Join([combo, r(np.pi, X_AXIS)])
                else:
                    combo = Join([combo, Combined([
                        Constant(1, 0, chan=r.chans[0]),
                        Constant(1, 0, chan=r.chans[1])
                    ])])

                combo = Join([combo, Combined([
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.readout_chan),
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.acq_chan),
                ])])

                s.append(combo)

        s = Sequencer(s)
        seqs = s.render()
        #seqs = join_all_small_elements(seqs)

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
#        s.plot_seqs(seqs)

        self.seqs=seqs
        return seqs

    def analyze(self, data=None, ax=None):
        ZS, pax = self.get_ys_ax(data, True)
        ret = analysis(self.delay_times, self.displacements, ZS, pax, bgcor=self.bgcor, plotdx=self.plotdx, plotdy=self.plotdy)
        if self.title:
            plt.title(self.title)
        if self.saveas:
            plt.savefig(self.saveas)
