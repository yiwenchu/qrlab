import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
import h5py
import lmfit

def exp_decay(params, x, data):
    est = params['ofs'] + params['amplitude'] * np.exp(-x / params['tau'].value)
    return data - est

def double_exp_decay(params, x, data):
    est = params['ofs'].value + params['amplitude'].value * np.exp(-x / params['tau'].value) + params['amplitude2'].value * np.exp(-x / params['tau2'].value)
    return data - est


def analysis(meas, data=None, fig=None):
    ys, fig = meas.get_ys_fig(data, fig)
    xs = meas.delays

    fig.axes[0].plot(xs/1e3, ys, 'ks', ms=3)

    if meas.double_exp == False:
        params = lmfit.Parameters()
        params.add('ofs', value=np.min(ys))
        params.add('amplitude', value=np.max(ys))
        params.add('tau', value=xs[-1]/2.0, min=50.0)
        result = lmfit.minimize(exp_decay, params, args=(xs, ys))
        lmfit.report_fit(params)

        fig.axes[0].plot(xs/1e3, -exp_decay(params, xs, 0), label='Fit, tau = %.03f us'%(params['tau'].value/1000.))
        fig.axes[0].legend(loc=0)
        fig.axes[0].set_ylabel('Intensity [AU]')
        fig.axes[0].set_xlabel('Time [us]')
        fig.axes[1].plot(xs, exp_decay(params, xs, ys), marker='s')

    else:
        params = lmfit.Parameters()
        params.add('ofs', value=np.min(ys))
        params.add('amplitude', value=np.max(ys)/2.0)
        params.add('tau', value=xs[-1], min=50.0)
        params.add('amplitude2', value=np.max(ys)/2.0)
        params.add('tau2', value=xs[-1]/4.0, min=50.0)
        result = lmfit.minimize(double_exp_decay, params, args=(xs, ys))
        lmfit.report_fit(params)

        weight1 = params['amplitude'].value / (params['amplitude'].value + params['amplitude2'].value)*100
        weight2 = 100-weight1
        text = 'Fit, tau = %.03f us +/- %.03f us (%.01f%%)\n     tau2 = %.03f us +/- %.03f us (%.01f%%)'%(
                params['tau'].value/1000.0, params['tau'].stderr/1000.0, weight1, params['tau2'].value/1000.0, params['tau2'].stderr/1000.0, weight2)
        fig.axes[0].plot(xs/1e3, -double_exp_decay(params, xs, 0), label=text)
        fig.axes[0].legend(loc=0)
        fig.axes[0].set_ylabel('Intensity [AU]')
        fig.axes[0].set_xlabel('Time [us]')
        fig.axes[1].plot(xs, double_exp_decay(params, xs, ys), marker='s')

    fig.canvas.draw()
    return params

class T1Measurement_laser(Measurement1D):

    def __init__(self, qubit_info, delays, double_exp=False, seq=None, postseq=None,
                 inj_len = None, QP_delay = None, laser_voltage = None, atten=None, edgewidth = 0.0,  **kwargs):
        self.qubit_info = qubit_info
        self.delays = delays
        self.xs = delays / 1e3      # For plotting purposes
        self.double_exp = double_exp
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq
        self.QP_delay = QP_delay # can be negative to measure during injection
        self.inj_len = inj_len
        self.laser_voltage = laser_voltage
        self.atten = atten
        self.rise = 0.625*edgewidth  #reported rise time is the 10%-%90% rise time 
                                #-here everything is referenced to the middle of the rise
        self.marker_width = 200.0 #marker width to use to trigger function gen for laser pulse
        self.trig_delay = 300.0 #function generator triggering delay
        self.vpulse_len = self.inj_len + (2.0/laser_voltage)*2.0*self.rise + self.rise
                            #desired laser pulse length
                            # + time for voltage to rise to 2V (lasing threshold)
                            # + approximate laser rise time


        super(T1Measurement_laser, self).__init__(len(delays), infos=qubit_info, **kwargs)
        self.data.create_dataset('delays', data=delays)
        self.data.set_attrs(inj_len=self.inj_len)
        self.data.set_attrs(QP_delay=self.QP_delay)
        self.data.set_attrs(laser_voltage = self.laser_voltage)
        self.data.set_attrs(attenuation_dB = self.atten)
        self.data.set_attrs(rise_time = 2*self.rise)

        

    def generate(self):
        s = Sequence()
#        s.append(Constant(250, 0, chan=4))#why was this here?
        r = self.qubit_info.rotate
        
                
        
        self.pi_delay = (-self.marker_width + self.trig_delay) + self.vpulse_len + self.rise + self.QP_delay 

        if self.pi_delay <= 0.0:
            raise ValueError('QP_delay is too negative\n')        
        
        for i, dt in enumerate(self.delays):
            s.append(self.seq)
            
            if 1:#self.QP_delay <= 0:
#                pre_pad = 100e3 #allows measurement before injection
##                post_pad = self.QP_delay + dt + 10e3
#                trig_delay = 0.0 #function generator triggering delay
#                pi_delay = pre_pad + 1.0*self.rise + trig_delay + self.inj_len + self.QP_delay 
#                
#                if self.QP_delay < -(pre_pad + self.inj_len):
#                    raise ValueError('Pi pulse too long before start of injection - ran out of padding')
#                
#                s.append(Combined([
#                    Join([Delay(pre_pad), Constant(self.inj_len, 1, chan="1m2")]),
#                    Join([Delay(pi_delay), r(np.pi, 0), Delay(dt), self.get_readout_pulse()])
#                    ], align=ALIGN_LEFT,))


                
                
                s.append(self.seq)
                s.append(Constant(self.marker_width, 1, chan = "1m2"))
                s.append(Delay(self.pi_delay))
                s.append(r(np.pi, 0))
                s.append(Delay(dt))
    
                if self.postseq is not None:
                    s.append(self.postseq)
                s.append(self.get_readout_pulse())                
                
            else:
                s.append(self.seq)
                s.append(Constant(self.inj_len, 1, chan = "1m2"))
                s.append(Delay(self.QP_delay))
                s.append(r(np.pi, 0))
                s.append(Delay(dt))
    
                if self.postseq is not None:
                    s.append(self.postseq)
                s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()
        return seqs

    def analyze(self, data=None, fig=None):
        self.fit_params = analysis(self, data, fig)
        return self.fit_params['tau'].value, self.fit_params['tau'].stderr
