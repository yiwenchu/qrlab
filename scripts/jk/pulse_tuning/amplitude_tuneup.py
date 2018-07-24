import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
import copy
import mclient
import lmfit
from lib.math import fitter
from matplotlib import gridspec

FIT_PARABOLA    = 'PARABOLA'    # Fit a parabola (to determine min/max pos)

class Amplitude_Tuneup(Measurement1D):

    def __init__(self, qubit_info, relative_range=None, update_ins=False,
                 seq=None, postseq=None, repeat_pulse=10,
                 selective=False, r_axis=0.0, ptype='pi',
                 **kwargs):

        self.qubit_info = qubit_info
        self.ptype = ptype

        if relative_range == None:
            relative_range = np.linspace(0.90,1.1,21)

        self.xs = relative_range

        # get current pulse amplitude settings.
        self.init_pi_amp = self.qubit_info.pi_amp
        self.init_pi2_amp = self.qubit_info.pi2_amp
        if selective:
            self.init_pi_amp = self.qubit_info.pi_amp_selective
            self.init_pi2_amp = self.qubit_info.pi2_amp_selective

        self.update_ins = update_ins
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq
        self.repeat_pulse = repeat_pulse
        self.r_axis = r_axis
        self.selective = selective

        if ptype == 'pi':
            self.amps = self.init_pi_amp*relative_range
        elif ptype == 'pi2':
            self.amps = self.init_pi2_amp*relative_range

            if self.repeat_pulse % 2:
                raise Exception('For pi/2 pulses an even number is required')

        super(Amplitude_Tuneup, self).__init__(len(self.amps), infos=(qubit_info,), **kwargs)
        self.data.create_dataset('relative_range', data=relative_range)

        self.fit_type = FIT_PARABOLA

    def generate(self):
        if self.selective:
            r = self.qubit_info.rotate_selective
        else:
            r = self.qubit_info.rotate

        s = Sequence()

        for i, amp in enumerate(self.amps):
            s.append(self.seq)
            s.append(Repeat(r(0, self.r_axis, amp=amp), self.repeat_pulse))
            if self.postseq is not None:
                s.append(self.postseq)
            s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()

        return seqs

    def analyze(self, data=None, fig=None):
        data = self.get_ys(data)
        parb_pos = self.analyze_parabola(data=data, fig=fig, xlabel='Amplitude', ylabel='Signal')

        if not self.selective:
            txt = '%s: %d ns\n' % (self.qubit_info.rotation, self.qubit_info.w)
        else:
            txt = '%s: %d ns\n' % (self.qubit_info.rotation_selective, self.qubit_info.w_selective)

        if self.update_ins:
            if self.ptype == 'pi':
                print 'Setting qubit pi-rotation ampltiude to %.06f' % self.init_pi_amp*parb_pos

            if self.ptype == 'pi2':
                print 'Setting qubit pi2-rotation ampltiude to %.06f' % self.init_pi2_amp*parb_pos

            ins = mclient.instruments[self.qubit_info.insname]
            if self.selective and self.ptype == 'pi':
                ins.set_pi_amp_selective(self.init_pi_amp*parb_pos)

            elif self.selective and self.ptype == 'pi2':
                ins.set_pi2_amp_selective(self.init_pi2_amp*parb_pos)

            elif self.ptype == 'pi':
                ins.set_pi_amp(self.init_pi_amp*parb_pos)

            elif self.ptype == 'pi2':
                ins.set_pi2_amp(self.init_pi2_amp*parb_pos)

        return parb_pos

