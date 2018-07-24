import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
import h5py
import lmfit

ECHO_NONE       = 'NONE'
ECHO_HAHN       = 'HANN'
ECHO_CPMG       = 'CMPG'
ECHO_XY4        = 'XY4'
ECHO_XY8        = 'XY8'
ECHO_XY16       = 'XY16'

def exp_decay(params, x, data):
    est = params['ofs'] + params['amplitude'] * np.exp(-(x-200.0) / params['tau'].value)
    return data - est

def pseudo_exp_decay(params, x, data):
    est = params['ofs'].value + params['amplitude'].value * (1-params['pseudoness'].value) / (np.exp((x-200.0)/params['tau'].value)-params['pseudoness'].value)
    return data - est

def analysis(meas, data=None, fig=None, fit_start='auto', vg=0.0, ve=6.0):
    ys_temp, fig = meas.get_ys_fig(data, fig)
    xs_temp = meas.QP_delays/1e3

    # Sorting and plotting the raw data
    dictionary={}
    for i in range(len(xs_temp)):
        dictionary[xs_temp[i]]=ys_temp[i]
    xs=[]
    ys=[]
    keys = dictionary.keys()
    keys.sort()
    for k in keys:
        xs.append(k)
        if dictionary[k] >= ve:
            dictionary[k] = ve
        if dictionary[k] <= vg:
            dictionary[k] = vg
        ys.append(dictionary[k])
    xs=np.array(xs)
    ys=np.array(ys)
    fig.axes[0].lines.pop()
    fig.axes[0].plot(xs, ys, 'b-', lw=1)

    # Converting the raw data into quantities proportional to phase and ultimately frequency shift, and plot it
    dv = ve - vg
    vmid = (vg+ve)/2.0
    ys = vmid + np.arcsin((ys-vmid)/dv*2)*(dv/2)
    fig.axes[0].plot(xs, ys, 'ks', ms=3)

    # Dump some points at the beginning of the curve
    # if the number of points is not specified, dumping points until finding two consecutive points meeting certain criteria
    if fit_start == 'auto':
        i=5
        while i<len(ys)-1 and ((ys[i]>vg*1.8) or (ys[i+1]>vg*1.8)):
            i=i+1
        fit_start = i

    if len(ys)-fit_start > 5:
        xs=xs[fit_start:]
        ys=ys[fit_start:]

        params = lmfit.Parameters()
        params.add('tau', value=5000, min=0)
        params.add('ofs', value=max(ys))
        params.add('amplitude', value=max(ys)-min(ys))

        result = lmfit.minimize(exp_decay, params, args=(xs, ys))
        lmfit.report_fit(params)
        fig.axes[0].plot(xs, -exp_decay(params, xs, 0), label='Fit, tau = %.03f ms +/- %.03f ms'%(params['tau'].value/1000.0, params['tau'].stderr/1000.0))
        fig.axes[0].legend(loc=0)
        fig.axes[0].set_ylabel('Intensity [AU]')
        fig.axes[0].set_xlabel('Time [us]')
        fig.axes[1].plot(xs, exp_decay(params, xs, ys), marker='s')

        fig.canvas.draw()
    else:
        params = lmfit.Parameters()
        params.add('tau', value=500, min=0)
        params['tau'].stderr = 0
    return params

class QPdecayRamsey(Measurement1D):

    def __init__(self, qubit_info, T2_delay, detune, rep_time, meas_per_reptime=1, meas_per_QPinj=None, fit_start='auto', fit_end=None,
                 vg=0.04, ve=7.21, inj_len=10e3, echotype=ECHO_HAHN, necho=1, **kwargs):

        self.T2_delay = T2_delay
        self.detune = detune
        self.qubit_info = qubit_info
        self.meas_per_QPinj = meas_per_QPinj
        self.meas_per_reptime = meas_per_reptime
        self.rep_time = rep_time
        self.vg = vg
        self.ve = ve
        self.inj_len = inj_len
        self.echotype = echotype
        self.necho = necho

        self.n_sat_inj = 0
        if inj_len > rep_time-5000:
            while inj_len > rep_time:
                inj_len = inj_len-rep_time+5000
                self.n_sat_inj +=1
            if inj_len < 250:
                print "Injection pulse length happens to be at a bad value!!"

        n_points = meas_per_reptime*meas_per_QPinj
        QP_delay_step = rep_time / meas_per_reptime
        self.qdt_delays = [i*QP_delay_step for i in range(meas_per_reptime)]

        QP_delays= np.linspace(0, QP_delay_step*(n_points-1), n_points)+5000
        self.QP_delays = np.transpose(np.reshape(QP_delays, (-1, meas_per_reptime))).flatten()
        self.xs = self.QP_delays / 1e3      # For plotting purposes

        self.fit_start = fit_start  # The number of points we skip at the beginning for doing the fitting
        if fit_end == None:
            self.fit_end = len(self.QP_delays)
        elif fit_end <=0:
            self.fit_end = len(self.QP_delays)-fit_end
        else:
            self.fit_end = fit_end  # The last point (point index in integer number) used in the fitting

        super(QPdecayRamsey, self).__init__(len(self.QP_delays), infos=qubit_info, **kwargs)

        # Saving all kinds of attributes:
        self.data.create_dataset('QP_delays', data=self.QP_delays)
        self.data.set_attrs(T2_delay=self.T2_delay)
        self.data.set_attrs(inj_len=inj_len)
        self.data.set_attrs(vg=self.vg)
        self.data.set_attrs(ve=self.ve)
        self.data.set_attrs(rep_time=rep_time)

    def get_echo_pulse(self):
        r = self.qubit_info.rotate

        if self.echotype == ECHO_NONE:
            return None

        elif self.echotype == ECHO_HAHN:
            return r(np.pi, X_AXIS)

        elif self.echotype == ECHO_CPMG:
            return r(np.pi, Y_AXIS)

        elif self.echotype == ECHO_XY4:
            return Sequence([
                r(np.pi, X_AXIS),
                r(np.pi, Y_AXIS),
                r(np.pi, X_AXIS),
                r(np.pi, Y_AXIS),
            ])

        elif self.echotype == ECHO_XY8:
            return Sequence([
                r(np.pi, X_AXIS),
                r(np.pi, Y_AXIS),
                r(np.pi, X_AXIS),
                r(np.pi, Y_AXIS),

                r(np.pi, Y_AXIS),
                r(np.pi, X_AXIS),
                r(np.pi, Y_AXIS),
                r(np.pi, X_AXIS),
            ])

        elif self.echo == ECHO_XY16:
            return Sequence([
                r(np.pi, X_AXIS),
                r(np.pi, Y_AXIS),
                r(np.pi, X_AXIS),
                r(np.pi, Y_AXIS),

                r(np.pi, Y_AXIS),
                r(np.pi, X_AXIS),
                r(np.pi, Y_AXIS),
                r(np.pi, X_AXIS),

                r(-np.pi, X_AXIS),
                r(-np.pi, Y_AXIS),
                r(-np.pi, X_AXIS),
                r(-np.pi, Y_AXIS),

                r(-np.pi, Y_AXIS),
                r(-np.pi, X_AXIS),
                r(-np.pi, Y_AXIS),
                r(-np.pi, X_AXIS),
            ])

    def generate(self):

        s = Sequence()
        s.append(Constant(250, 0, chan=4))
        r = self.qubit_info.rotate
        inj_len = self.inj_len - (self.rep_time-5000)*self.n_sat_inj  #the part of injection pulse within one clock cycle

        e = self.get_echo_pulse()
        if e:
            elen = e.get_length()
            e = Pad(e, 250, PAD_BOTH)
            epadlen = e.get_length() - elen
        else:
            elen = 0

        for j, qdt in enumerate(self.qdt_delays):

            s.append(Trigger(dt=250))
            inj_delay = self.rep_time-inj_len-5000
            s.append(Delay(inj_delay)) # wait for a delay time before injection so that the end of injection lines up with the next trigger

            # fire the injection pulse, led by an additional pi pulse that may make injection more effective
            s.append(r(np.pi, 0))
            if inj_len < 20000:
                s.append(Constant(inj_len, 1, chan="3m2"))
            else:  # if injection pulse is longer than 20us, we break it into 10us segments:
                n_10us_pulse = int(inj_len)/10000-1
                s.append(Repeat(Constant(10000, 1, chan="3m2"), n_10us_pulse))
                s.append(Constant(inj_len-n_10us_pulse*10000, 1, chan="3m2"))

            # If the injection pulse is even longer than the rep time, we pre-fire the pulse for n_sat_inj clock cycles:
            for x in range(self.n_sat_inj):
#                s.append(Constant(250, 0, chan=4))
                s.append(Trigger(250))
                s.append(r(np.pi, 0))
                n_10us_pulse = int(self.rep_time-5000)/10000 - 1
                s.append(Repeat(Constant(10000, 1, chan="3m2"), n_10us_pulse))
                s.append(Constant(self.rep_time-5000-n_10us_pulse*10000, 1, chan="3m2"))

            for i in range(self.meas_per_QPinj):
                s.append(Trigger(dt=250))
                s.append(Delay(qdt))
                s.append(r(np.pi/2, X_AXIS))
                if e:
                    tau = int(np.round(self.T2_delay / (2 * self.necho) - epadlen/2))
                    if tau < 0:
                        s.append(Delay(self.T2_delay))
                    else:
                        s.append(Delay(tau))
                        for i in range(self.necho - 1):
                            s.append(e)
                            s.append(Delay(2*tau))
                        s.append(e)
                        s.append(Delay(tau))
                else:# Plain T2
                    s.append(Delay(self.T2_delay))
                angle = self.T2_delay * 1e-9 * self.detune * 2 * np.pi
                s.append(r(np.pi/2, angle))
                s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs


    def analyze(self, data=None, fig=None):
        self.fit_params = analysis(self, data, fig, self.fit_start, self.vg, self.ve)
        return self.fit_params['tau'].value
