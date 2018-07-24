# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 15:42:10 2013

@author: Brian Vlastakis / Reinier Heeres
"""

import numpy as np
import matplotlib.pyplot as plt

from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement2D, STYLE_LINES
from lib.math import fit

PLOT_ANGLE  = 'ANGLE'
PLOT_RE     = 'RE'
PLOT_ABS    = 'ABS'

def analysis(meas, data=None, fig=None):
    zs, fig = meas.get_ys_fig(data, fig)
    plt.xlabel('T [ns]')
    plt.ylabel('Intensity')
    fig.canvas.draw()

class RotFrameTest(Measurement2D):

    def __init__(self, qubit_info, cav_info, delays, disps, disp0=0, detune=0, bgcor=False, plot_type=PLOT_ANGLE, seq=None, **kwargs):
        self.qubit_info = qubit_info
        self.cav_info = cav_info
        self.bgcor = bgcor
        self.delays = delays
        self.disps = np.array(disps)
        self.detune = detune
        self.plot_type = plot_type

        # Initial sequence
        if seq is None:
            if disp0 != 0:
                seq = Join([Trigger(250), cav_info.rotate(np.abs(disp0), np.angle(disp0))])
            else:
                seq = Trigger(250)
        self.seq = seq

        # Set plotting parameters
        self.xs = delays
        if plot_type == PLOT_ANGLE:
            self.ys = np.angle(disps)
        elif plot_type == PLOT_ABS:
            self.ys = np.abs(disps)
        elif plot_type == PLOT_RE:
            self.ys = np.real(disps)

        npoints = len(delays) * len(disps)
        if bgcor:
            npoints *= 2
        super(RotFrameTest, self).__init__(npoints, infos=(qubit_info, cav_info), style=STYLE_LINES, residuals=False, **kwargs)
        self.data.create_dataset('delays', data=self.delays)
        self.data.create_dataset('displacements', data=self.disps, dtype=self.disps.dtype)
        self.data.set_attrs(
            disp0=disp0,
        )

    def generate(self):
        '''
        If bg = True generate background measurement, i.e. without qubit pi pulse
        '''

        s = Sequence()

        r = self.qubit_info.rotate
        c = self.cav_info.rotate
        for i_delay, delay in enumerate(self.delays):
            for i_disp, disp in enumerate(self.disps):
                for i_bg in range(2):
                    if i_bg == 1 and not self.bgcor:
                        continue

                    s.append(self.seq)
                    if delay > 0:
                        s.append(Delay(delay))

                    if self.detune != 0:
                        dphi = 2 * np.pi * delay * 1e-9 * self.detune
                    else:
                        dphi = 0
                    add = c(np.abs(disp), np.angle(disp) + np.pi + dphi)
                    if i_bg == 0:
                        add = Join([add, r(np.pi, X_AXIS)])
                    else:
                        add = Join([add, Combined([
                            Constant(1, 0, chan=r.chans[0]),
                            Constant(1, 0, chan=r.chans[1])
                        ])])

                    add = Join([add, Combined([
                        Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.readout_chan),
                        Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.acq_chan),
                    ])])

                    s.append(add)

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def get_ys(self, data=None):
        ys = super(RotFrameTest, self).get_ys(data)
        if self.bgcor:
            return ys[::2] - ys[1::2]
        return ys

    def analyze(self, data=None, fig=None):
        ret = analysis(self, data, fig)
