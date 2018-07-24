import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
import matplotlib.pyplot as plt
import copy
import mclient
import lmfit

def fit_timerabi(params, x, data):
    decay = np.exp(-x / params['decay'].value)
    est = params['ofs'].value - params['amp'].value * np.cos(2*np.pi*x / params['period'].value) * decay
    return data  - est

def analysis(meas, data=None, fig=None, period=None):
    ys, fig = meas.get_ys_fig(data, fig)
    xs = meas.amps

    fig.axes[0].plot(xs, ys, 'ks', ms=3)

    amp0 = (np.max(ys) - np.min(ys)) / 2
#    if ys[0]>np.average(ys):
#        amp0 = -amp0
    fftys = np.abs(np.fft.fft(ys - np.average(ys)))
    fftfs = np.fft.fftfreq(len(ys), xs[1]-xs[0])
    period0 = 1 / np.abs(fftfs[np.argmax(fftys)])

    params = lmfit.Parameters()
    params.add('ofs', value=np.average(ys))
    params.add('amp', value=amp0)
    if period is not None:
        params.add('period', value=period, min=0, vary=False)
    else:
        params.add('period', value=period0, min=0)
    params.add('decay', value=1e9, min=0, vary=False)
    result = lmfit.minimize(fit_timerabi, params, args=(xs, ys))
    lmfit.report_fit(params)

    txt = 'Amp = %.03f +- %.03e\nPeriod = %.03f +- %.03e\nPi amp = %.05f' % (params['amp'].value, params['amp'].stderr, params['period'].value, params['period'].stderr, params['period'].value/2 )
    fig.axes[0].plot(xs, -fit_timerabi(params, xs, 0), label=txt)
    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[0].set_xlabel('Pulse area')
    fig.axes[0].legend(loc=0)

    fig.axes[1].plot(xs, fit_timerabi(params, xs, ys), marker='s')
    fig.canvas.draw()
    return params

class EFRabi(Measurement1D):

    def __init__(self, ge_info, ef_info, amps, first_pi=True, second_pi=True,
                 update=False, seq=None, extra_info=None, laser_power = None,
                 force_period=None,
                 **kwargs):
        self.ge_info = ge_info
        self.ef_info = ef_info
        self.amps = amps
        self.first_pi = first_pi
        self.second_pi = second_pi
        self.xs = amps
        self.update_ins = update
        self.force_period = force_period
        self.laser_power = laser_power
        if seq is None:
            seq = Trigger(250)
        self.seq = seq

        super(EFRabi, self).__init__(len(amps), infos=(ge_info, ef_info), **kwargs)
        self.data.create_dataset('areas', data=amps)
        if laser_power:
            self.data.set_attrs(laser_power =self.laser_power)

    def generate(self):
        s = Sequence()

        for i, amp in enumerate(self.amps):
            r = self.ge_info.rotate
            r_ef = self.ef_info.rotate
            add = self.seq
            if self.first_pi:
                add = Join([Trigger(dt=250), r(np.pi, 0), Delay(5)])
            add = Join([add, r_ef(0, 0, amp = amp), Delay(5)])
            if self.second_pi:
                add = Join([add, r(np.pi, 0)])
#            marker_switch = Constant(, 1, chan=self.)
            s.append(add)
            s.append(Combined([
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.readout_chan),
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.acq_chan),
                ]))

        s = self.get_sequencer(s)
        seqs = s.render()
        self.seqs = seqs
        return seqs

    def analyze(self, data=None, fig=None):
        self.fit_params = analysis(self, data=data, fig=fig, period=self.force_period)
        pi_area = self.fit_params['period'].value / 2
        if self.update_ins:
            print 'Setting qubit pi-rotation area to %.03f' % pi_area
            mclient.instruments[self.ef_info.insname].set_pi_area(pi_area)
        return pi_area
