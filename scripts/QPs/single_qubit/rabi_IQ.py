import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
import matplotlib.pyplot as plt
import copy
import mclient
import lmfit

gen = mclient.instruments['gen']

def fit_timerabi(params, x, data):
    decay = np.exp(-x / params['decay'].value)
    est = params['ofs'].value - params['amp'].value * np.cos(2*np.pi*x / params['period'].value + params['phase'].value) * decay
    return data  - est

def analysis(meas, data=None, fig=None):
    ys, fig = meas.get_ys_fig(data, fig)
    xs = meas.areas

    fig.axes[0].plot(xs, ys, 'ks', ms=3)

    amp0 = (np.max(ys) - np.min(ys)) / 2
    if ys[0]>np.average(ys):
        amp0 = -amp0
    fftys = np.abs(np.fft.fft(ys - np.average(ys)))
    fftfs = np.fft.fftfreq(len(ys), xs[1]-xs[0])
    period0 = 1 / np.abs(fftfs[np.argmax(fftys)])

    params = lmfit.Parameters()
    params.add('ofs', value=np.average(ys))
    params.add('amp', value=amp0)
    if meas.fix_period is not None:
        params.add('period', value=meas.fix_period, vary=False)
    else:
        params.add('period', value=period0, min=0)
    params.add('decay', value=20000, min=0)
    params.add('phase', value=0, min=-np.pi, max=np.pi, vary=(not meas.fix_phase))

    result = lmfit.minimize(fit_timerabi, params, args=(xs, ys))
    lmfit.report_fit(params)

    txt = 'Amp = %.03f +- %.03e\nPeriod = %.03f +- %.03e\nPi area = %.03f' % (params['amp'].value, params['amp'].stderr, params['period'].value, params['period'].stderr, params['period'].value/2 )
    fig.axes[0].plot(xs, -fit_timerabi(params, xs, 0), label=txt)
    fig.axes[0].set_ylabel('Intensity [AU]')
    fig.axes[0].set_xlabel('Pulse area')
    fig.axes[0].legend(loc=0)

    fig.axes[1].plot(xs, fit_timerabi(params, xs, ys), marker='s')
    fig.canvas.draw()
    return params

class Rabi(Measurement1D):

    def __init__(self, qubit_info, areas, update=False, seq=None, fix_phase=True, fix_period=None, repeat_pulse=1, postseq=None, **kwargs):
        self.qubit_info = qubit_info
        self.areas = areas
        self.xs = areas
#        self.plen = plen
        self.update_ins = update
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq
        self.fix_phase = fix_phase
        self.fix_period = fix_period
        self.repeat_pulse = repeat_pulse

#        super(Rabi, self).__init__(len(areas), infos= qubit_info, **kwargs)
#        self.data.create_dataset('area', data=areas)

        super(Rabi, self).__init__(len(areas), infos= qubit_info, **kwargs)
        self.data.create_dataset('area', data=areas)

    def generate(self):
        s = Sequence()
#        s.append(Constant(250, 1, chan='4m1'))
        s.append(Constant(250, 0, chan=4))

        for i, pulse_area in enumerate(self.areas):
            s.append(self.seq)

            r = copy.deepcopy(self.qubit_info.rotate)
            r.set_pi(pulse_area)
            rpi = r(np.pi, 0)
            s.append(rpi)

            if self.postseq is not None:
                s.append(self.postseq)

            s.append(Combined([
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.readout_chan),
                    Constant(self.readout_info.pulse_len, 1, chan=self.readout_info.acq_chan),
                ]))


        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def analyze(self, data=None, fig=None):
        self.fit_params = analysis(self, data=data, fig=fig)
        pi_area = self.fit_params['period'].value / 2
        if self.update_ins:
            print 'Setting qubit pi-rotation area to %.03f' % pi_area
            mclient.instruments[self.qubit_info.insname].set_pi_area(pi_area)
        return pi_area
