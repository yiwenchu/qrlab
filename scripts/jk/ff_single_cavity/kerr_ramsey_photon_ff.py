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

from scripts.single_cavity import kerr_ramsey_photon as KRP

def analysis(phases, delays, data, fig):
    return KRP.analysis(phases, delays, data, fig)

class KerrRamseyPhoton_FF(KRP.KerrRamseyPhoton):

    def __init__(self, cav_info_01, cav_info_12, qubit_info, phases, delays,
                 flux_chan=1, flux_amp=0.0, seq=None, postseq=None,
                 subtraction=False,
                 **kwargs):

        self.cav_info_01 = cav_info_01
        self.cav_info_12 = cav_info_12
        self.qubit_info = qubit_info

        self.phases = phases
        self.delays = delays
        self.xs = phases / (2 * np.pi)

        self.flux_chan = flux_chan
        self.flux_amp = flux_amp

        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq
        self.infos = [cav_info_01, cav_info_12, qubit_info,]
        self.subtraction = subtraction

        super(KerrRamseyPhoton_FF, self).__init__(cav_info_01, cav_info_12, qubit_info,
                    phases, delays, seq=seq, postseq=postseq, subtraction=subtraction, **kwargs)

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
                    s.append(Constant(delay, self.flux_amp, chan=self.flux_chan))
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
                    s.append(Delay(Constant(delay, self.flux_amp, chan=self.flux_chan))
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
