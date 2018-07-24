import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
from lib.math import fitter

def analysis(meas, data=None, fig=None, txt = ''):
    ys, fig = meas.get_ys_fig(data, fig)
    xs = meas.detunings
    #####
    fit = fitter.Fitter('lorentzian')
    p = fit.get_lmfit_parameters(xs, ys)
    result = fit.perform_lmfit(xs, ys, print_report=False, plot=False, p=p)
    datafit = fit.eval_func()

    x0 = result.params['x0']
    fit_label = txt+'center = %0.4f MHz,' % (x0/1e6) + ' FWHM = %0.4f MHz' % (result.params['w']/1e6)
    ######

    fig.axes[0].plot(xs/1e6, ys, marker='s', ms=3, label='')
    fig.axes[0].plot(xs/1e6, datafit, ls='-', ms=3, label=fit_label) #####

    # fig.axes[0].plot(xs/1e6, ys)
    fig.axes[0].set_xlabel('Detuning (MHz)')
    fig.axes[0].set_ylabel('Intensity (AU)')
    # fig.canvas.draw()
    fig.axes[0].legend(loc='best')

    return result.params

class SSBSpec_fit(Measurement1D):

    def __init__(self, qubit_info, detunings, seq=None, seq2=None, simulseq = None, postseq = None, bgcor=False, txt = '', **kwargs):
        self.qubit_info = qubit_info
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.seq2 = seq2
        self.simulseq = simulseq
        self.postseq = postseq        
        self.detunings = detunings
        self.xs = detunings / 1e6       # For plot
        self.bgcor = bgcor
        self.txt = txt

        npoints = len(detunings)
        if bgcor:
            npoints += 2
        super(SSBSpec_fit, self).__init__(npoints, residuals=False, infos=(qubit_info,), **kwargs)
        self.data.create_dataset('detunings', data=detunings)

    def generate(self):
        s = Sequence()

        ro = Combined([
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.readout_chan),
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.acq_chan),
        ])

        if self.bgcor:
            r = self.qubit_info.rotate
#            plen = self.qubit_info.rotate.base(np.pi, 0).get_length()
            s.append(self.seq)
            s.append(ro)            

            s.append(self.seq)
            s.append(r(np.pi, X_AXIS))
            s.append(ro)

        for i, df in enumerate(self.detunings):
            g = DetunedSum(self.qubit_info.rotate.base, self.qubit_info.w_selective, chans=self.qubit_info.sideband_channels)
            if df != 0:
                period = 1e9 / df
            else:
                period = 1e50
            g.add(self.qubit_info.pi_amp_selective, period)

            s.append(self.seq)
            s.append(self.seq2)

            if self.simulseq is not None:
                s.append(Combined((g(), self.simulseq), align = 2))
            else: 
                s.append(g())
            
            if self.postseq is not None:
                s.append(self.postseq)
            
            s.append(ro)

        s = self.get_sequencer(s)
        s.add_marker('3m1', 3, ofs = -85, bufwidth = 5) #a hack for now to output marker for stark shift pulse
        seqs = s.render()
        return seqs

    def get_ys(self, data=None):
        ys = super(SSBSpec_fit, self).get_ys(data)
        if self.bgcor:
            return (ys[2:]-ys[0])/(ys[1]-ys[0])
#            return ys[1:] - ys[0]
        return ys

    def analyze(self, data=None, fig=None):
        self.fit_params = analysis(self, data, fig, txt = self.txt)
        return self.fit_params ####


