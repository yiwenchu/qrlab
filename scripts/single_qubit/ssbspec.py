import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D

def analysis(meas, data=None, fig=None):
    ys, fig = meas.get_ys_fig(data, fig)
    xs = meas.detunings
    fig.axes[0].plot(xs/1e6, ys)
    fig.axes[0].set_xlabel('Detuning (MHz)')
    fig.axes[0].set_ylabel('Intensity (AU)')
    fig.canvas.draw()

class SSBSpec(Measurement1D):

    def __init__(self, qubit_info, detunings, seq=None, postseq = None, bgcor=False, **kwargs):
        self.qubit_info = qubit_info
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq        
        self.detunings = detunings
        self.xs = detunings / 1e6       # For plot
        self.bgcor = bgcor
        self.selective = kwargs.get('selective', False)
        if not self.selective:
            self.pi_pulse = self.qubit_info.rotate
            self.pi_amp = self.qubit_info.pi_amp
            self.w = self.qubit_info.w
        else:
            self.pi_pulse = self.qubit_info.rotate_selective
            self.pi_amp = self.qubit_info.pi_amp_selective
            self.w = self.qubit_info.w_selective
            

        npoints = len(detunings)
        if bgcor:
            npoints += 1
        super(SSBSpec, self).__init__(npoints, residuals=False, infos=(qubit_info,), **kwargs)
        self.data.create_dataset('detunings', data=detunings)

    def generate(self):
        s = Sequence()

        ro = Combined([
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.readout_chan),
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.acq_chan),
        ])

        if self.bgcor:
            plen = self.pi_pulse.base(np.pi, 0).get_length()
            s.append(self.seq)
            s.append(Delay(plen))
            s.append(ro)

        for i, df in enumerate(self.detunings):
            g = DetunedSum(self.pi_pulse.base, self.w, chans=self.qubit_info.sideband_channels)
            if df != 0:
                period = 1e9 / df
            else:
                period = 1e50
            g.add(self.pi_amp, period)

            s.append(self.seq)
            s.append(g())
            
            if self.postseq is not None:
                s.append(self.postseq)
            
            s.append(ro)

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def get_ys(self, data=None):
        ys = super(SSBSpec, self).get_ys(data)
        if self.bgcor:
            return ys[1:] - ys[0]
        return ys

    def analyze(self, data=None, fig=None):
        analysis(self, data, fig)
