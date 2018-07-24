import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D
import h5py
import lmfit

def exp_decay(params, x, data):
    est = params['ofs'] + params['amplitude'] * np.exp(-(x-1000.0) / params['tau'].value)
    return data - est

def log_exp_decay(params, x, data):
    est = np.log(params['ofs'].value + params['amplitude'].value * np.exp(-(x-1000.0) / params['tau'].value))/np.log(10)
    return data - est

def pseudo_exp_decay(params, x, data):
    est = params['ofs'].value + params['amplitude'].value * (1-params['pseudoness'].value) / (np.exp((x-1000.0)/params['tau'].value)-params['pseudoness'].value)
    return data - est

def log_pseudo_exp_decay(params, x, data):
    est = np.log(params['ofs'].value + params['amplitude'].value * (1-params['pseudoness'].value) / (np.exp((x-1000.0)/params['tau'].value)-params['pseudoness'].value))/np.log(10)
    return data - est

def tanh_decay(params, x, data):
    est = data
    return data - est

def analysis(meas, data=None, fig=None, fit_start='auto', fit_end='auto', vg=-0.25, ve=9.2, eff_T1_delay=400.0, tolerance=1e-4):
    ys_temp, fig = meas.get_ys_fig(data, fig)
    xs_temp = meas.QP_delays/1e3
    if meas.extraT1_delays is not None:
        t1exp_temp = ys_temp[-len(meas.extraT1_delays):]
        ys_temp = ys_temp[:-len(meas.extraT1_delays)]
    dv=ve-vg
    if len(fig.axes[0].lines)>0:
        fig.axes[0].lines.pop()
    else:
        params = lmfit.Parameters()
        params.add('tau', value=500, min=0)
        params['tau'].stderr = 0
        params.add('ofs', value=0.1, min=0)
        return params

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
        if dictionary[k] <= vg:
            dictionary[k] = vg+dv*0.01
        elif dictionary[k] >= ve:
            dictionary[k] = vg+dv*0.99
        ys.append(dictionary[k])
    xs=np.array(xs)
    ys=np.array(ys)
    fig.axes[0].plot(xs, ys, 'b-', lw=1)

    if meas.smart_T1_delay == False:
        # Converting the raw data into quantities proportional to qubit decay rate, and plot it
        ys = ve - dv*np.log(dv/(ys-vg))
        fig.axes[0].plot(xs, ys, 'ks', ms=3)
        # Dump some points at the beginning of the curve
        # if the number of points is not specified, dumping points until finding two consecutive points meeting certain criteria
        if fit_start == 'auto':
            i=5
            while i<len(ys)-1 and ((ys[i]>vg*1.8) or (ys[i+1]>vg*1.8)):
                i=i+1
            fit_start = i
        if len(ys)-fit_start > 8:
            xs=xs[fit_start:]
            ys=ys[fit_start:]

            params = lmfit.Parameters()
            params.add('tau', value=2000, min=0)
            params.add('ofs', value=max(ys))
            params.add('amplitude', value=-max(ys))

            result = lmfit.minimize(exp_decay, params, args=(xs, ys))
            lmfit.report_fit(params)
            fig.axes[0].plot(xs, -exp_decay(params, xs, 0), label='Fit, tau = %.03f ms +/- %.03f ms'%(params['tau'].value/1000.0, params['tau'].stderr/1000.0))
    #        fig.axes[0].legend(loc=0)
            fig.axes[0].set_ylabel('Intensity [AU]')
            fig.axes[0].set_xlabel('Time [us]')
            fig.axes[1].plot(xs, exp_decay(params, xs, ys), marker='s')

            params2 = lmfit.Parameters()
            params2.add('tau', value=params['tau'].value, min=0)
            params2.add('ofs', value=params['ofs'].value)
            params2.add('amplitude', value=-max(ys), vary=True)
            params2.add('pseudoness', value=0.001, vary=True, min=0.0001, max=0.9999)
            result = lmfit.minimize(pseudo_exp_decay, params2, args=(xs,ys))
            lmfit.report_fit(params2)

            pseudo = params2['pseudoness'].value
            pseudo_err = params2['pseudoness'].stderr
            tau_QP = params2['tau'].value/1000.0
            tau_QP_err = params2['tau'].stderr/1000.0
            text = 'Fit, tau = %.03f ms +/- %.03f ms\npseudoness = %.03f +/- %.03f' %(tau_QP, tau_QP_err, pseudo, pseudo_err)
            fig.axes[0].plot(xs, -pseudo_exp_decay(params2, xs, 0), label=text)
            fig.axes[0].legend(loc=0)

            fig.axes[1].plot(xs, pseudo_exp_decay(params2, xs, ys), marker='^')
            fig.canvas.draw()
        else:
            params = lmfit.Parameters()
            params.add('tau', value=500, min=0)
            params['tau'].stderr = 0

    else:
        ys = np.log(dv/(ys-vg)) / (meas.T1_delays+eff_T1_delay)*1000.0  # Should work even if vg>ve
        meas.QP_delays_sorted = xs
        meas.invT1 = copy.copy(ys)
        meas.log_invT1 = np.log(ys)/np.log(10)
        fig.axes[0].plot(xs, meas.invT1, 'm^', ms=4)

        if fit_start == 'auto':
            fit_start = 0
        if fit_end == 'auto':
            fit_end = len(ys)
        xs=meas.QP_delays_sorted[fit_start:fit_end]
        ys=meas.invT1[fit_start:fit_end]
#        log_ys=meas.log_invT1_adj[fit_start:fit_end]
        log_ys=meas.log_invT1[fit_start:fit_end]

        params = lmfit.Parameters()
        params.add('tau', value=xs[-1]/4.0, min=0)
        params.add('ofs', value=min(ys))
        params.add('amplitude', value=max(ys))
        result = lmfit.minimize(log_exp_decay, params, args=(xs, log_ys))
        lmfit.report_fit(params)
        text = 'Fit, tau = %.03f ms +/- %.03f ms\nT1-floor = %.02f us +/- %.02f us' %(params['tau'].value/1000.0, params['tau'].stderr/1000.0, 1/params['ofs'].value, params['ofs'].stderr/(params['ofs'].value**2))
        fig.axes[0].plot(xs, -exp_decay(params, xs, 0), label=text)
        fig.axes[0].set_ylabel('Qubit Relaxation rate (1/us)')
        fig.axes[0].set_xlabel('Time [us]')
        fig.axes[1].plot(xs, log_exp_decay(params, xs, log_ys), marker='s')

        params2 = lmfit.Parameters()
        params2.add('tau', value=params['tau'].value, min=0)
        params2.add('ofs', value=params['ofs'].value)
        params2.add('amplitude', value=params['amplitude'].value, vary=True)
        params2.add('pseudoness', value=0.001, vary=True, min=0.0001, max=0.9999)
        result = lmfit.minimize(log_pseudo_exp_decay, params2, args=(xs,log_ys))
        lmfit.report_fit(params2)

        pseudo = params2['pseudoness'].value
        pseudo_err = params2['pseudoness'].stderr
        tau_QP = params2['tau'].value/1000.0
        tau_QP_err = params2['tau'].stderr/1000.0
        Gamma_ref = params2['amplitude'].value/(1-pseudo)
#        Gamma_ref_err = params2['amplitude'].stderr/(1-pseudo)+params2['amplitude'].value*(1-pseudo_err)/(1-pseudo)
        Delta = 180e-6*1.6e-19/1.056e-34
        fq = 7.6e9
        C = np.sqrt(2*fq*Delta/np.pi)
        inv_r = tau_QP*(1-pseudo)/pseudo*Gamma_ref/C*1e12
        text = 'Fit, tau = %.03f ms +/- %.03f ms\npseudoness = %.03f +/- %.03f\nT1-floor = %.02f us +/- %.02f us' %(tau_QP, tau_QP_err, pseudo, pseudo_err, 1/params2['ofs'].value, params2['ofs'].stderr/(params2['ofs'].value**2))
        text2 = '\nrecomb. constant: 1/(%.03f ns)'%(inv_r)

        fig.axes[0].plot(xs, -pseudo_exp_decay(params2, xs, 0), label=text+text2)
        fig.axes[0].legend(loc=0)
        fig.axes[1].plot(xs, log_pseudo_exp_decay(params2, xs, log_ys), marker='^')
        fig.axes[0].set_yscale('log')

        offguess = params2['ofs'].value
        ys_fudged = copy.copy(ys)
        for i, y in enumerate(ys_fudged):
            if y-offguess < tolerance:
                ys_fudged[i] = offguess + tolerance
        fig.axes[0].plot(xs, ys_fudged-offguess, 'ks', ms=3)
        params_temp = copy.deepcopy(params2)
        params_temp['ofs'].value = 0.0
        fig.axes[0].plot(xs, -pseudo_exp_decay(params_temp, xs, 0))
        fig.axes[0].set_xlim(0, max(xs_temp))
        fig.canvas.draw()

        meas.xs_fit = xs
        meas.ys_fit = -exp_decay(params, xs, 0)
        meas.log_ys_fit = -log_exp_decay(params, xs, 0)

        if meas.extraT1_delays is not None:
            fig2 = plt.figure()
            plt.plot(meas.extraT1_delays/1e3, t1exp_temp)
            params = lmfit.Parameters()
            params.add('ofs', value=np.min(t1exp_temp))
            params.add('amplitude', value=np.max(t1exp_temp))
            params.add('tau', value=20e3, min=50.0)
            result = lmfit.minimize(exp_decay, params, args=(meas.extraT1_delays, t1exp_temp))
            lmfit.report_fit(params)
            fig2.axes[0].plot(meas.extraT1_delays/1e3, -exp_decay(params, meas.extraT1_delays, 0), label='Fit, tau = %.03f us'%(params['tau'].value/1000.))
            fig2.axes[0].legend(loc=0)
            fig2.axes[0].set_ylabel('Intensity [AU]')
            fig2.axes[0].set_xlabel('Time [us]')

    return params2

class QPdecay_laser(Measurement1D):

    def __init__(self, qubit_info, T1_delays, rep_time, meas_per_reptime=1, meas_per_QPinj=None, fit_start='auto', fit_end=None, vg=0.04, ve=7.21, eff_T1_delay=2000.0, inj_len=10e3, atten=None, laser_voltage=None, edgewidth = 1e3, extraT1_delays=None, **kwargs):

        if type(T1_delays) is np.ndarray: # This means we are doing variable T1_delays designated by the T1_delay array.
            self.smart_T1_delay = True
            meas_per_QPinj = len(T1_delays)/meas_per_reptime
            self.T1_delays = T1_delays
            self.T1_delays_2D = np.transpose(np.reshape(T1_delays, (-1, meas_per_reptime))) # Reshaping the T1_delay array to match the nature of our sequence generation
        else:
            self.smart_T1_delay = False
            self.T1_delay = T1_delays

        self.qubit_info = qubit_info
        self.meas_per_QPinj = meas_per_QPinj
        self.meas_per_reptime = meas_per_reptime
        self.rep_time = rep_time
        self.vg = vg
        self.ve = ve
        self.eff_T1_delay = eff_T1_delay
        self.inj_len = inj_len
        self.extraT1_delays = extraT1_delays
        self.atten = atten
        self.laser_voltage = laser_voltage

        self.rise = 0.625*edgewidth  #reported rise time is the 10%-%90% rise time 
                                #-here everything is referenced to the middle of the rise
        self.marker_width = 200.0 #marker width to use to trigger function gen for laser pulse
        self.trig_delay = 300.0 #function generator triggering delay
        self.vpulse_len = self.inj_len + (2.0/laser_voltage)*2.0*self.rise + self.rise
                            #desired laser pulse length
                            # + time for voltage to rise to 2V (lasing threshold)
                            # + approximate laser rise time

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
        if self.extraT1_delays is not None:
            self.xs = np.concatenate((self.QP_delays/1e3, self.extraT1_delays+max(self.QP_delays/1e3)))      # For plotting purposes
        else:
            self.xs = self.QP_delays / 1e3      # For plotting purposes

        self.fit_start = fit_start  # The number of points we skip at the beginning for doing the fitting
        if fit_end == None:
            self.fit_end = len(self.QP_delays)
        elif fit_end <=0:
            self.fit_end = len(self.QP_delays)-fit_end
        else:
            self.fit_end = fit_end  # The last point (point index in integer number) used in the fitting

        if self.extraT1_delays is not None:
            super(QPdecay_laser, self).__init__(len(self.QP_delays)+len(self.extraT1_delays), infos=qubit_info, **kwargs)
        else:
            super(QPdecay_laser, self).__init__(len(self.QP_delays), infos=qubit_info, **kwargs)

        # Saving all kinds of attributes:
        self.data.create_dataset('QP_delays', data=self.QP_delays)
        if self.smart_T1_delay == True:
            self.data.create_dataset('T1_delays', data=self.T1_delays)
            self.T1_delay = 'vary'
        self.data.set_attrs(T1_delay=self.T1_delay)
        self.data.set_attrs(inj_len=self.inj_len)
        self.data.set_attrs(attenuation_dB=self.atten)
        self.data.set_attrs(laser_voltage=self.laser_voltage)
        self.data.set_attrs(vg=self.vg)
        self.data.set_attrs(ve=self.ve)
        self.data.set_attrs(eff_T1_delay=self.eff_T1_delay)
        self.data.set_attrs(rep_time=rep_time)
        self.data.set_attrs(rise_time = 2*self.rise)


    def generate(self):

        s = Sequence()
        s.append(Constant(250, 0, chan=4))
        r = self.qubit_info.rotate
        inj_len = self.inj_len - (self.rep_time-5000)*self.n_sat_inj  #the part of injection pulse within one clock cycle



        for j, qdt in enumerate(self.qdt_delays):

            s.append(Trigger(dt=250))
            inj_delay = self.rep_time-self.vpulse_len-2000
            s.append(Delay(inj_delay)) # wait for a delay time before injection so that the end of injection lines up with the next trigger

            #trigger the function generator to start the laser pulse, followed by a delay before starting the measurement sequence
            s.append(Constant(self.marker_width, 1, chan = "1m2"))
#            if inj_len < 20000:
#                s.append(Constant(inj_len, 1, chan="3m2"))
#            else:  # if injection pulse is longer than 20us, we break it into 10us segments:
#                n_10us_pulse = int(inj_len)/10000-1
#                s.append(Repeat(Constant(10000, 1, chan="3m2"), n_10us_pulse))
#                s.append(Constant(inj_len-n_10us_pulse*10000, 1, chan="3m2"))

            # If the injection pulse is even longer than the rep time, we pre-fire the pulse for n_sat_inj clock cycles:
#            for x in range(self.n_sat_inj):
##                s.append(Constant(250, 0, chan=4))
#                s.append(Trigger(250))
#                s.append(r(np.pi, 0))
#                n_10us_pulse = int(self.rep_time-5000)/10000 - 1
#                s.append(Repeat(Constant(10000, 1, chan="3m2"), n_10us_pulse))
#                s.append(Constant(self.rep_time-5000-n_10us_pulse*10000, 1, chan="3m2"))


            if self.smart_T1_delay is True: # This means we are doing variable T1_delays designated by the T1_delay array.
                for dt in self.T1_delays_2D[j]:
                    s.append(Trigger(250))
                    s.append(Delay(qdt))
                    s.append(r(np.pi, 0))
                    s.append(Delay(dt))
                    s.append(self.get_readout_pulse())

            else:  # Fixed T1 delay, rarely used now.
                for i in range(self.meas_per_QPinj):
                    s.append(Trigger(250))
                    s.append(Delay(qdt))
                    s.append(r(np.pi, 0))
                    s.append(Delay(self.T1_delay))
                    s.append(self.get_readout_pulse())

        if self.extraT1_delays is not None:
            for i, dt in enumerate(self.extraT1_delays):
                s.append(Trigger(250))
                s.append(r(np.pi, 0))
                s.append(Delay(dt))
                s.append(self.get_readout_pulse())

        s = self.get_sequencer(s)
        seqs = s.render()

        return seqs

    def analyze(self, data=None, fig=None):
        self.fit_params = analysis(self, data, fig, self.fit_start, self.fit_end, self.vg, self.ve, self.eff_T1_delay)
        return self.fit_params['tau'].value
