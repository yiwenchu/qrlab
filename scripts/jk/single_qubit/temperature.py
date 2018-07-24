import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
import copy
import mclient
import lmfit
# import pulseseq.sequencer_subroutines as ss
from lib.math import fitter

FIT_AMP         = 'AMP'         # Fit simple sine wave
FIT_AMPFUNC     = 'AMPFUNC'     # Try to fit amplitude curve based on pi/2 and pi amp
FIT_PARABOLA    = 'PARABOLA'    # Fit a parabola (to determine min/max pos)

'''
    TODO:  Constrain the fit to use the frequency/phase from the stronger
        signal.
            Done, but it's automatically the first dataset that sets the frequency
'''
def analysis(meas, data=None, fig=None, **kwargs):
    #to constrain a fit params, add kwarg 'var_name' = value
    if 'dphi' not in kwargs:
        kwargs['dphi'] = -np.pi/2
        
    ys, fig = meas.get_ys_fig(data, fig)
    ys = data
    xs = meas.amps

    fig.axes[0].plot(xs, ys, 'ks', ms=3)

    if meas.fit_type == FIT_AMPFUNC:

        pass

    else: # meas.fit_type == FIT_AMP

        r = fitter.Fitter('sine')
        result = r.perform_lmfit(xs, ys, print_report=False, plot=False, **kwargs)
        p = result.params
        # ys_fit = r.test_values(xs, p=p)
        ys_fit = r.eval_func(xs, params=p)

        pi_amp = 1 / (2.0 * p['f'].value)

        txt = 'Amp = %0.3f +/- %0.4f\n' % (p['A'].value, p['A'].stderr)
        txt += 'f = %0.4f +/- %0.5f\n' % (p['f'].value, p['A'].stderr)
        txt += 'phase = %0.3f +/- %0.4f\n' % (p['dphi'].value, p['dphi'].stderr)
        fig.axes[0].plot(xs, ys_fit, label=txt)
        fig.axes[1].plot(xs, ys-ys_fit, marker='s')

    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[0].set_xlabel('Pulse amplitude')
    fig.axes[0].legend(loc=0)

    fig.canvas.draw()
    return result.params

class Temperature(Measurement1D):

    def __init__(self, ef_info, ge_info, amps, update=False,
                 seq=None, r_axis=0, fix_phase=False,
                 fix_period=None, postseq=None,
                 fit_type=FIT_AMP,
                 **kwargs):

        self.ge_info = ge_info
        self.ef_info = ef_info
        self.amps = amps
        self.xs = amps

        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq
        self.fix_phase = fix_phase
        self.fix_period = fix_period
        self.r_axis = r_axis
        self.fit_type = fit_type
        self.infos = [ge_info, ef_info]

        super(Temperature, self).__init__(2*len(amps), infos=self.infos, **kwargs)
        self.data.create_dataset('amps', data=amps)

    def generate(self):
        r_ge = self.ge_info.rotate
        r_ef = self.ef_info.rotate
#
        s = Sequence()

        for i, amp in enumerate(self.amps):
            s.append(self.seq)

            s.append(r_ge(np.pi,0))
            s.append(r_ef(0, 0, amp=amp))
            s.append(r_ge(np.pi,0))

            if self.postseq is not None:
                s.append(self.postseq)
            s.append(self.get_readout_pulse())


        for i, amp in enumerate(self.amps):
            s.append(self.seq)

            s.append(Constant(r_ge(np.pi,0).get_length(),0,chan='foo'))
            s.append(r_ef(0, 0, amp=amp))
            s.append(r_ge(np.pi,0))

            if self.postseq is not None:
                s.append(self.postseq)
            s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()

        return seqs

    def analyze(self, data=None, fig=None):
        data1 = data[:len(self.amps)]
        self.fit_params = analysis(self, data=data1, fig=fig)
        amp1 = self.fit_params['A'].value
        f1 = self.fit_params['f'].value

        data2 = data[len(self.amps):]
        self.fit_params = analysis(self, data=data2, fig=fig, f=f1)
        amp2 = self.fit_params['A'].value

        print 'Polarization ~ %0.03f' % (amp2/amp1, )
        print '|e> population ~ %0.03f' % (amp2/(amp1+amp2), )

    def update(self, avg_data):
        ys = self.get_ys(avg_data)
        ys1 = ys[:len(self.amps)]
        ys2 = ys[len(self.amps):]

        fig = self.get_figure()
        fig.axes[0].clear()
        if hasattr(self, 'xs'):
            fig.axes[0].plot(self.xs, ys1)
            fig.axes[0].plot(self.xs, ys2)
        else:
            fig.axes[0].plot(ys1)
            fig.axes[0].plot(ys2)
        fig.canvas.draw()

        return 1

