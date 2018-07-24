# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D_bgcor
from lib.math import fitter
import lmfit

def analysis(meas, data=None):
    if data is None:
        data = meas.get_ys()

    if len(meas.xs) > 1 and len(meas.ys) >1:
        centers = []
        data = data.reshape((len(meas.xs), len(meas.ys)))

        for xidx in range(len(meas.xs)):
            fit = fitter.Fitter('gaussian')
            result = fit.perform_lmfit(meas.ys, data[xidx],
                                       plot=False,
                                       print_report=True)
            centers.append(result.params['x0'].value)

        plt.figure()
        plt.plot(meas.xs, centers)

        fit = fitter.Fitter('linear')
        result = fit.perform_lmfit(meas.xs, centers, plot=False)
        p = result.params
        ys_fit = fit.test_values(meas.xs, p=p)

        plt.plot(meas.xs, ys_fit)

        detuning_MHz = 1e3*result.params['m'].value/(2*np.pi)
        plt.title('Detuning: %0.6f MHz' % detuning_MHz)
        plt.show()

    return detuning_MHz

class ChiTimeDomain(Measurement1D_bgcor):

    def __init__(self, qubit_info, cav_info, delays, alpha0 = 1.0,
                 bgcor=True, seq=None, r_axis=0.0, **kwargs):
        self.qubit_info = qubit_info
        self.cav_info = cav_info
        self.bgcor = bgcor
        self.delays = np.array(delays)
        self.alpha0 = alpha0
        self.r_axis = r_axis

        # Initial sequence
        self.seq = seq

        # Set plotting parameters
        self.xs = self.delays

        npoints = len(delays)
        if bgcor:
            npoints *= 2

        super(ChiTimeDomain, self).__init__(npoints, infos=(qubit_info, cav_info),
              bgcor=bgcor, **kwargs)
        self.data.create_dataset('delays', data=self.delays)
        self.data.set_attrs(
            alpha0=alpha0,
            bgcor=bgcor,
        )

    def generate(self):
        '''
        If bg = True generate background measurement, i.e. without qubit pi pulse
        '''

        s = Sequence()

        r = self.qubit_info.rotate
        c = self.cav_info.rotate
        for i_delay, delay in enumerate(self.delays):
            for i_bg in [1,0]:
                if i_bg == 1 and not self.bgcor:
                    continue

                if self.seq is None:
                    s.append(Trigger(250))
                else:
                    # seq should have a trigger
                    s.append(self.seq)

                s.append(c(np.pi * np.abs(self.alpha0), np.angle(self.alpha0)))
                s.append(r(np.pi/2, self.r_axis))
#                s.append(Combined([
#                            c(np.pi * np.abs(self.alpha0), np.angle(self.alpha0)),
#                            r(np.pi/2, self.r_axis),
#                        ]))

                if delay > 0:
                    s.append(Delay(delay))

                s.append(r(np.pi/2, self.r_axis + np.pi*i_bg))

#                s.append(r(np.pi*i_bg, 0))
                s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def analyze(self, data=None, fig=None):
        pass
        #self.detuning = analysis(self, self.get_ys())


