# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:42:10 2013

@author: Brian Vlastakis
"""

import numpy as np
import matplotlib.pyplot as plt

from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement2D
from lib.math import fit


def analysis(distance, quadrature, data, ax=None, bgcor=False, plotdx=None, plotdy = None):
    if bgcor:
        data = data[::2] - data[1::2]

    XS = distance
    YS = np.abs(quadrature)
    ZS = data.reshape(XS.shape)

    if plotdx is not None:
        plt.subplots(nrows = 1, ncols = 1)
        plt.pcolormesh(plotdx, plotdy, ZS)
    else:
        plt.pcolormesh(XS, YS, ZS)
    ax = plt.gca()
    plt.colorbar()

    ax.set_xlim(np.min(plotdx), np.max(plotdx))
    ax.set_ylim(np.min(plotdy), np.max(plotdy))
    ax.set_xlabel(r'$Distance \{\beta \}$')
    ax.set_ylabel(r'$Quadrature \{\alpha \}$')

class TomoCuts(Measurement2D):

    def __init__(self, qubit_info, cav_info, dmax, Nd, bmax, Nb, chi_delay = 0, qubit_rot = np.pi,bgcor=False, extra_info=None,saveas=None, **kwargs):
        self.qubit_info = qubit_info
        self.cav_info = cav_info
        self.bgcor = bgcor
        self.qubit_rot = np.abs(qubit_rot)
        self.qubit_angle = np.angle(qubit_rot)
        self.chi_delay = chi_delay
        self.extra_info = extra_info
        self.saveas = saveas

        xs = np.linspace(0, dmax, Nd)
        deltax = (xs[1] - xs[0])/2
        self.plotdx = np.linspace(0-deltax, dmax+deltax, Nd+1)#np.linspace(0-delta, tmax+delta, Nt+1)

        ys = np.linspace(-bmax, bmax, Nb)
        ys_plot = np.abs(ys)
        deltay = (ys_plot[1] - ys_plot[0])/2
        self.plotdy = np.linspace(-np.abs(bmax)-deltay, np.abs(bmax)+deltay, Nb+1)


        XS, YS = np.meshgrid(xs, ys)
        self.distance = XS
        self.quadrature = YS


        npoints = len(xs)*len(ys)
        if bgcor:
            npoints *= 2
        super(TomoCuts, self).__init__(npoints, **kwargs)
        self.data.create_dataset('distance', data=self.distance, dtype=np.float)
        #self.data.create_dataset('quadrature', data=self.quadrature, dtype=np.complex)

    def generate(self):

        '''
        If bg = True generate background measurement, i.e. without qubit pi pulse
        '''

        s = Sequence()

        r = self.qubit_info.rotate
        c = self.cav_info.rotate
        distance_list = self.distance.flatten()
        quadrature_list = self.quadrature.flatten()
        npoints = len(distance_list)

        for i in range(npoints):
            for i_bg in range(2):
                if i_bg == 1 and not self.bgcor:
                    continue

                s.append(Trigger(250))
                s.append(r(np.pi/2, X_AXIS))

                beta = distance_list[i]/2  #divide by two since we will create cats with distance 2*alpha
                alpha = quadrature_list[i]
                dispPulseIni = c(np.abs(beta), 0)
                dispQuadrature = c(-np.abs(alpha), np.angle(alpha))
                #s.append(dispPulse)

                delayPulse = Constant(self.chi_delay, 0.0, chan=self.cav_info.channels[0])
                #delayPulse = smartConstant(delay_time, [0], chans = self.cav_info.channels)
                combo = Join([dispPulseIni,delayPulse,dispQuadrature])
                if i_bg ==0:
                    combo = Join([combo, r(self.qubit_rot, self.qubit_angle)])
                else:
                    combo = Join([combo, r(self.qubit_rot - np.pi, self.qubit_angle)])

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
        ZS, pax = self.get_ys_ax(data, ax)
        ret = analysis(self.distance, self.quadrature, ZS, pax, bgcor=self.bgcor, plotdx=self.plotdx, plotdy=self.plotdy)
        if self.title:
            plt.title(self.title)
        if self.saveas:
            plt.savefig(self.saveas)
