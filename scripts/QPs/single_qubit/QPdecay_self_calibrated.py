import numpy as np
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from measurement import Measurement1D_bgproc
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

def sort_data(meas, data=None):
        # Sorting the raw data
        xs_temp = meas.QP_delays/1e3
        ys_max, ys_mid, ys_min = meas.get_all_data(data)

        dict_max = {}
        dict_mid = {}
        dict_min = {}


        for i in range(len(xs_temp)):
            dict_max[xs_temp[i]]=ys_max[i]
            dict_mid[xs_temp[i]]=ys_mid[i]
            dict_min[xs_temp[i]]=ys_min[i]
          
        xs=[]
        ys_max=[]
        ys_mid=[]
        ys_min=[]
        keys = dict_max.keys()
        keys.sort()
        for k in keys:
            xs.append(k)
    #        if dictionary[k] <= vg:
    #            dictionary[k] = vg+dv*0.01
    #        elif dictionary[k] >= ve:
    #            dictionary[k] = vg+dv*0.99
            ys_max.append(dict_max[k])
            ys_mid.append(dict_mid[k])
            ys_min.append(dict_min[k])
            
        xs=np.array(xs)
        ys_max=np.array(ys_max)
        ys_mid=np.array(ys_mid)
        ys_min=np.array(ys_min)

        return xs, ys_max, ys_mid, ys_min

def analysis(meas, data=None, fig=None, fit_start='auto', fit_end='auto', vg=-0.25, ve=9.2, tolerance=1e-4):    
    ys_temp, fig = meas.get_ys_fig(data, fig)
    xs, ys = meas.process_data()
    # xs = xs/1e3
    # if meas.extraT1_delays is not None:
    #     t1exp_temp = ys_temp[-len(meas.extraT1_delays):]
    #     ys_temp = ys_temp[:-len(meas.extraT1_delays)]

    if len(fig.axes[0].lines)>0:
        pass
        # fig.axes[0].lines.pop()
    else:
        params = lmfit.Parameters()
        params.add('tau', value=500, min=0)
        params['tau'].stderr = 0
        params.add('ofs', value=0.1, min=0)
        return params

    fig.axes[1].plot(xs, ys, 'b-', lw=1)


    if meas.smart_T1_delay == False:
        pass
        #not supported but not used
        # Converting the raw data into quantities proportional to qubit decay rate, and plot it
    #     ys = ve - dv*np.log(dv/(ys-vg))
    #     fig.axes[0].plot(xs, ys, 'ks', ms=3)
    #     # Dump some points at the beginning of the curve
    #     # if the number of points is not specified, dumping points until finding two consecutive points meeting certain criteria
    #     if fit_start == 'auto':
    #         i=5
    #         while i<len(ys)-1 and ((ys[i]>vg*1.8) or (ys[i+1]>vg*1.8)):
    #             i=i+1
    #         fit_start = i
    #     if len(ys)-fit_start > 8:
    #         xs=xs[fit_start:]
    #         ys=ys[fit_start:]

    #         params = lmfit.Parameters()
    #         params.add('tau', value=2000, min=0)
    #         params.add('ofs', value=max(ys))
    #         params.add('amplitude', value=-max(ys))

    #         result = lmfit.minimize(exp_decay, params, args=(xs, ys))
    #         lmfit.report_fit(params)
    #         fig.axes[0].plot(xs, -exp_decay(params, xs, 0), label='Fit, tau = %.03f ms +/- %.03f ms'%(params['tau'].value/1000.0, params['tau'].stderr/1000.0))
    # #        fig.axes[0].legend(loc=0)
    #         fig.axes[0].set_ylabel('Intensity [AU]')
    #         fig.axes[0].set_xlabel('Time [us]')
    #         fig.axes[1].plot(xs, exp_decay(params, xs, ys), marker='s')

    #         params2 = lmfit.Parameters()
    #         params2.add('tau', value=params['tau'].value, min=0)
    #         params2.add('ofs', value=params['ofs'].value)
    #         params2.add('amplitude', value=-max(ys), vary=True)
    #         params2.add('pseudoness', value=0.001, vary=True, min=0.0001, max=0.9999)
    #         result = lmfit.minimize(pseudo_exp_decay, params2, args=(xs,ys))
    #         lmfit.report_fit(params2)

    #         pseudo = params2['pseudoness'].value
    #         pseudo_err = params2['pseudoness'].stderr
    #         tau_QP = params2['tau'].value/1000.0
    #         tau_QP_err = params2['tau'].stderr/1000.0
    #         text = 'Fit, tau = %.03f ms +/- %.03f ms\npseudoness = %.03f +/- %.03f' %(tau_QP, tau_QP_err, pseudo, pseudo_err)
    #         fig.axes[0].plot(xs, -pseudo_exp_decay(params2, xs, 0), label=text)
    #         fig.axes[0].legend(loc=0)

    #         fig.axes[1].plot(xs, pseudo_exp_decay(params2, xs, ys), marker='^')
    #         fig.canvas.draw()
    #     else:
    #         params = lmfit.Parameters()
    #         params.add('tau', value=500, min=0)
    #         params['tau'].stderr = 0

    else:
        meas.QP_delays_sorted = xs
        meas.invT1 = copy.copy(ys)
        meas.log_invT1 = np.log(ys)/np.log(10)
        fig.axes[1].clear()
        fig.axes[1].plot(xs, meas.invT1, 'm^', ms=4)

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
        fig.axes[1].plot(xs, -exp_decay(params, xs, 0), label=text)
        fig.axes[1].set_ylabel('Qubit Relaxation rate (1/us)')
        fig.axes[1].set_xlabel('Time [us]')
        fig.axes[2].plot(xs, log_exp_decay(params, xs, log_ys), marker='s')

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

        fig.axes[1].plot(xs, -pseudo_exp_decay(params2, xs, 0), label=text)#+text2)
        fig.axes[1].legend(loc=0)
        fig.axes[2].plot(xs, log_pseudo_exp_decay(params2, xs, log_ys), marker='^')
        fig.axes[1].set_yscale('log')

        offguess = params2['ofs'].value
        ys_fudged = copy.copy(ys)
        for i, y in enumerate(ys_fudged):
            if y-offguess < tolerance:
                ys_fudged[i] = offguess + tolerance
        # fig.axes[1].plot(xs, ys_fudged-offguess, 'ks', ms=3)
        params_temp = copy.deepcopy(params2)
        params_temp['ofs'].value = 0.0
        fig.axes[1].plot(xs, -pseudo_exp_decay(params_temp, xs, 0))
        fig.axes[1].set_xlim(0, max(xs))
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
            fig2.axes[1].plot(meas.extraT1_delays/1e3, -exp_decay(params, meas.extraT1_delays, 0), label='Fit, tau = %.03f us'%(params['tau'].value/1000.))
            fig2.axes[1].legend(loc=0)
            fig2.axes[1].set_ylabel('Intensity [AU]')
            fig2.axes[1].set_xlabel('Time [us]')

    return params2

class QPdecay_self_calibrated(Measurement1D_bgproc):

    def __init__(self, qubit_info, T1_delays, rep_time, meas_per_reptime=1, meas_per_QPinj=None, fit_start='auto', fit_end=None, vg=0.04, ve=7.21, eff_T1_delay=2000.0, inj_len=10e3, extraT1_delays=None, **kwargs):
        
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
        self.marker_chan = kwargs.get('injection_marker',"3m2")
        if extraT1_delays is not None:
            print 'Extra delays not supported!'
        self.extraT1_delays = None

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
            super(QPdecay_self_calibrated, self).__init__(len(self.QP_delays)+len(self.extraT1_delays), infos=qubit_info, **kwargs)
        else:
            super(QPdecay_self_calibrated, self).__init__(3*len(self.QP_delays), infos=qubit_info, **kwargs)

        # Saving all kinds of attributes:
        self.data.create_dataset('QP_delays', data=self.QP_delays)
        if self.smart_T1_delay == True:
            self.data.create_dataset('T1_delays', data=self.T1_delays)
            self.T1_delay = 'vary'
        self.data.set_attrs(T1_delay=self.T1_delay)
        self.data.set_attrs(inj_len=inj_len)
        self.data.set_attrs(vg=self.vg)
        self.data.set_attrs(ve=self.ve)
        self.data.set_attrs(eff_T1_delay=self.eff_T1_delay)
        self.data.set_attrs(rep_time=rep_time)

    def get_all_data(self, data=None):
        ys = self.get_ys(data)
        if not self.bgproc:
            return ys

        mpqinj = self.meas_per_QPinj
        mprep = self.meas_per_reptime
        base_slice = np.array(range(mpqinj))
        full_slice = np.array([], dtype=int)
        for j in range(mprep):
            full_slice = np.concatenate((full_slice, base_slice + 3*j*mpqinj))


        ys_max = ys[full_slice]
        ys_mid = ys[full_slice + mpqinj]
        ys_min = ys[full_slice + 2*mpqinj]
#           
        return ys_max, ys_mid, ys_min
    

    def process_data(self, data=None):
        #returns processed data (qubit relaxation rate) in time ordered
        xs, ys_max, ys_mid, ys_min = sort_data(self)

        ys = np.log((ys_max - ys_min)/(ys_mid-ys_min)) / (self.T1_delays)*1000.0 
        return xs, ys
   
    def update(self, avg_data):
        super(QPdecay_self_calibrated, self).update(avg_data)
        
        xs, max_data, mid_data, min_data = sort_data(self)
        fig = self.get_figure()
        fig.axes[0].clear()
        fig.axes[1].clear()

        xs, ys = self.process_data()

        if hasattr(self, 'xs'):
            fig.axes[0].plot(xs, max_data, 'ks', label='no pi delay')
            fig.axes[0].plot(xs, mid_data, 'bs', label='pi delay')
            fig.axes[0].plot(xs, min_data, 'gs', label='no pi pulse')
            fig.axes[1].plot(xs, ys, 'ks-', label='relaxation rate')
        else:
            fig.axes[0].plot(max_data, 'k-', label='no pi delay')
            fig.axes[0].plot(mid_data, 'b-', label='pi delay')
            fig.axes[0].plot(min_data, 'g-', label='no pi pulse')
            fig.axes[1].plot(ys, 'ks-', label='relaxation rate')

        fig.axes[1].semilogy()
        fig.axes[0].legend(loc='best')
        fig.axes[1].legend(loc='best')

        fig.canvas.draw()

    def generate(self):

        s = Sequence()
#        s.append(Constant(250, 0, chan=4))
        r = self.qubit_info.rotate
        inj_len = self.inj_len - (self.rep_time-5000)*self.n_sat_inj  #the part of injection pulse within one clock cycle

        
        for j, qdt in enumerate(self.qdt_delays):
            
            for pt in ['max', 'mid', 'floor']:
                pi_delay = 1
                pi_pulse = 1
                if pt == 'max':
                    #calibrate maximum voltage value
                    pi_delay = 0
                if pt == 'floor':
                    #calibrate minimum voltage value
                    pi_pulse = 0

    
                s.append(Trigger(250))
                inj_delay = self.rep_time-inj_len-5000
                s.append(Delay(inj_delay)) # wait for a delay time before injection so that the end of injection lines up with the next trigger
    
                # fire the injection pulse, led by an additional pi pulse that may make injection more effective
                s.append(r(np.pi, 0))
                if inj_len < 20000:
                    s.append(Constant(inj_len, 1, chan=self.marker_chan))
                else:  # if injection pulse is longer than 20us, we break it into 10us segments:
                    n_10us_pulse = int(inj_len)/10000-1
                    s.append(Repeat(Constant(10000, 1, chan=self.marker_chan), n_10us_pulse))
                    s.append(Constant(inj_len-n_10us_pulse*10000, 1, chan=self.marker_chan))
    
                # If the injection pulse is even longer than the rep time, we pre-fire the pulse for n_sat_inj clock cycles:
                for x in range(self.n_sat_inj):
    #                s.append(Constant(250, 0, chan=4))
                    s.append(Trigger(250))
                    s.append(r(np.pi, 0))
                    n_10us_pulse = int(self.rep_time-5000)/10000 - 1
                    s.append(Repeat(Constant(10000, 1, chan=self.marker_chan), n_10us_pulse))
                    s.append(Constant(self.rep_time-5000-n_10us_pulse*10000, 1, chan=self.marker_chan))
    
    
                if self.smart_T1_delay is True: # This means we are doing variable T1_delays designated by the T1_delay array.
                    for dt in self.T1_delays_2D[j]:
                        s.append(Trigger(250))
                        s.append(Delay(qdt))
                        s.append(r(pi_pulse*np.pi, 0))
                        s.append(Delay(pi_delay*dt))
                        s.append(self.get_readout_pulse())
                        s.append(Delay((1 - pi_delay)*dt))
    
                else:  # Fixed T1 delay, rarely used now.
                    for i in range(self.meas_per_QPinj):
                        s.append(Trigger(250))
                        s.append(Delay(qdt))
                        s.append(r(pi_pulse*np.pi, 0))
                        s.append(Delay(pi_delay*self.T1_delay))
                        s.append(self.get_readout_pulse())
                        s.append(Delay((1 - pi_delay)*self.T1_delay))
                        
    
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
        self.fit_params = analysis(self, data, fig, self.fit_start, self.fit_end)
        return self.fit_params['tau'].value
