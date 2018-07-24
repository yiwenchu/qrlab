from measurement import Measurement1D
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from lib.math import fitter
import time
import objectsharer as objsh
import numpy as np

SPEC   = 0
POWER  = 1

class Tracked_Spectroscopy(Measurement1D):
    '''

    '''

    def __init__(self, qubit_info,
                 qubit_yoko,
                 source_type, #choices are CURR or VOLT
                 set_vals,
                 init_freqs, # program will handle updating frequency range
                 spec_params, # format (qubit_rfsource, init_spec_power)
                 init_ro_power,
                 init_ro_freq,
                 ro_range = 2, #will search += xdbm, 0 if not doing power tuning 
                 ro_step = 1,
                 ro_freq_tune = 1, #tune freq every x experiments; -1 if using frequency interval only;
                 ro_pwr_tune = 1,
                 ro_freq_tune_interval = 100e6, #tune ro frequency if the qubit frequency has changed by this much since last tuning, -1 if using steps only
                 ro_pwr_tune_interval = 100e6, #tune ro pwr if the qubit frequency has changed by this much since last tuning, -1 if using steps only
                 ro_shots = 1e4,
                 ro_spec_range = 10e6, #range for spec when retuning readout, 0 if not doing frequency tuning
                 width_min=2e6, #Pulled these numbers out of rear.  plz fiz.
                 width_max=6e6,
                 freq_step=0e6,
                 plen=50e3, amp=1, seq=None, postseq=None,
                 pow_delay=1, freq_delay=0.1, plot_type=None,
                 use_IQge=0, use_weight=0,
                 subtraction=0,
                 **kwargs):

        self.qubit_info = qubit_info
        self.qubit_yoko = qubit_yoko
        self.source_type = source_type

        self.freqs = init_freqs
        self.num_freq_pts = len(init_freqs)
        self.freq_range = max(init_freqs)-min(init_freqs)

        self.qubit_rfsource, self.spec_power = spec_params
        self.width_min = width_min #for adjusting spec power
        self.width_max = width_max

        self.freq_step = freq_step # frequency step estimator,
                                   # Set to 0 if you want constant frequency

        self.ro_shots = ro_shots
        self.ro_power = init_ro_power #for adjusting RO power
        self.ro_freq = init_ro_freq
        self.ro_range = ro_range
        self.ro_step = ro_step

        self.ro_pwr_tune = ro_pwr_tune
        self.ro_freq_tune = ro_freq_tune
        self.ro_freq_tune_interval = ro_freq_tune_interval
        self.ro_pwr_tune_interval = ro_pwr_tune_interval
        self.ro_spec_range = ro_spec_range

        self.set_vals = set_vals

        self.use_IQge = use_IQge
        self.use_weight = use_weight
        self.subtraction = subtraction

        self.num_steps = len(self.set_vals)

        self.plen = plen
        self.amp = amp
        self.pow_delay = pow_delay
        self.freq_delay = freq_delay
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.postseq = postseq

        self.plot_type = plot_type

        self.extra_info = None

        super(Tracked_Spectroscopy, self).__init__(1, infos=qubit_info, **kwargs)
        self.ro_source = self.readout_info.rfsource1
        self.ro_LO = self.readout_info.rfsource2

        self.yoko_set_vals = self.data.create_dataset('yoko_set_vals', shape=[self.num_steps])
        self.ro_pwr_log = self.data.create_dataset('ro_powers', shape=[self.num_steps])
        self.ro_freq_log = self.data.create_dataset('ro_freqs', shape=[self.num_steps])
        self.spec_log = self.data.create_dataset('spec_powers', shape=[self.num_steps])
        self.freqs_log = self.data.create_dataset('freqs', shape=[self.num_steps, self.num_freq_pts])
        self.center_freqs =  self.data.create_dataset('center_freqs', shape=[self.num_steps])
        self.widths =  self.data.create_dataset('widths', shape=[self.num_steps])

        self.IQs = self.data.create_dataset('avg', shape=[self.num_steps, self.num_freq_pts], dtype=np.complex)
        self.amps = self.data.create_dataset('avg_pp', shape=[self.num_steps, self.num_freq_pts])

    def measure(self):
        from scripts.single_qubit import spectroscopy
        from scripts.calibration import ropower_calibration
        from scripts.calibration import ro_power_cal
        from scripts.single_cavity import rocavspectroscopy

        spec = lambda q_freqs, ro_power, spec_power, yoko_set_val: \
            spectroscopy.Spectroscopy(
                                       self.qubit_info,
                                       q_freqs,
                                       [self.qubit_rfsource, spec_power],
                                       # [ro_power],
                                       plen=self.plen,
                                       amp=self.amp,
                                       seq=self.seq,
                                       postseq=self.postseq,
                                       # pow_delay=self.pow_delay,
                                       extra_info=self.extra_info,
                                       freq_delay=self.freq_delay,
                                       plot_type=self.plot_type,
                                       title='Pulsed spectroscopy: (ro, spec) = (%0.2f, %0.2f) dBm; yoko %0.3e V\n' % (ro_power, spec_power, yoko_set_val),
                                       analyze_data=True,
                                       keep_data=True,
                                       use_IQge=self.use_IQge,
                                       use_weight=self.use_weight,
                                       subtraction=self.subtraction)

        ropcal = lambda powers: ropower_calibration.ROPower_Calibration(
                self.qubit_info,
                powers,
                qubit_pulse=ropower_calibration.SAT_PULSE)

        orcpcal = lambda powers, curAmp: ro_power_cal.Optimal_Readout_Power(
                self.qubit_info,
                powers,
                plen=self.plen, amp=curAmp, update_readout=True, verbose_plots=False,
                plot_best=False,
                shots=self.ro_shots)

        rospec = lambda power, ro_freqs: rocavspectroscopy.ROCavSpectroscopy(self.qubit_info, [power], ro_freqs)

        # initialize yoko to a known set_val
        if self.source_type == 'VOLT':
            self.qubit_yoko.do_set_voltage(0)
        elif self.source_type == 'CURR':
            self.qubit_yoko.do_set_current(0)
        else:
            print 'source type needs to be VOLT or CURR'
            return False
        self.qubit_yoko.set_output_state(1)

        time.sleep(1)
        self.last_center = np.average(self.freqs) # middle point
        self.last_tune_center = np.average(self.freqs)
        self.ro_source.set_frequency(self.ro_freq)

        for idx, set_val in enumerate(self.set_vals):
            if self.source_type == 'VOLT':
                self.qubit_yoko.do_set_voltage(set_val)
            if self.source_type == 'CURR':
                self.qubit_yoko.do_set_current(set_val)

            time.sleep(1)
            # run spec at this voltage
            # print [self.ro_power, self.spec_power, self.plen]

            # tune ro frequency
            if ((self.ro_freq_tune != -1 and (idx % self.ro_freq_tune) == 0)\
                or (self.ro_freq_tune == -1 and np.absolute(self.last_center-self.last_tune_center) > self.ro_freq_tune_interval))\
                and (self.ro_spec_range !=0):

                ro_freqs = np.linspace(self.ro_freq - self.ro_spec_range, 
                                        self.ro_freq + self.ro_spec_range, 
                                        51)
                ro = rospec(self.ro_power, ro_freqs)
                ro.measure()
                plt.close()

                self.ro_freq = ro.fit_params[0][2]
                self.ro_source.set_frequency(self.ro_freq)
                self.ro_LO.set_frequency(self.ro_freq+50e6)

#                # update power
#                self.ro_power = r.get_best_power()
                # self.ro_power = r.best_power
                self.last_tune_center = self.last_center

            spec_exp = spec(self.freqs, self.ro_power, self.spec_power, set_val)
            spec_exp.measure()
            plt.close()

            center = spec_exp.fit_params['x0'].value * 1e6
            width = spec_exp.fit_params['w'].value * 1e6

            #save RO power, spec power, freqs, center, width, raw data..
            # self.IQs[idx,:] = spec_exp.IQs[0,:]
            # self.amps[idx,:] = spec_exp.amps[0,:]

            self.IQs[idx,:] = spec_exp.IQs[:]
            self.amps[idx,:] = spec_exp.reals[:]      

            # save spec fit and experiment parameter data
            self.center_freqs[idx] = center
            self.widths[idx] = width
            self.yoko_set_vals[idx] = set_val
            self.ro_pwr_log[idx] = self.ro_power
            self.ro_freq_log[idx] = self.ro_freq
            self.spec_log[idx] = self.spec_power
            self.freqs_log[idx,:] = self.freqs

            #choose the next frequency range
            delta = center - self.last_center
            if self.freq_step == 0:
                # no step, keep the same frequency range
                self.freqs = self.freqs
                self.last_center = center
            elif idx == 0: # first point; don't try to stop
                self.freqs = self.freqs + delta + self.freq_step
                self.last_center = center
            else:
                # use 0th order estimator from last points
                # we will weight the frequency step to the expected direction
                # of the frequency step
                if self.freq_step > 0:
                    # increasing freq
                    ll, ul = np.array([-0.25, 1.25]) * self.freq_step
                else:
                    # decreasing freq
                    ll, ul = np.array([1.25, -0.25]) * self.freq_step

                #Commented out 9/3/14 Jacob, not quite right.
                # if not ll < delta < ul:
                #     print 'quitting: we seem to have lost the qubit'
                #     break

                #if the width is off, probably a sign that we're not fitting to qubit peak
                if not self.width_min*0.5 < width < self.width_max*2:
                    print 'quitting: we seem to have lost the qubit'
                    break

                # pick new frequency range such that the NEXT point should be centered
                q_freqs_center = np.average(self.freqs)

                # current point is centered
                self.freqs = self.freqs + (center - q_freqs_center)
#                blah = np.average(self.freqs)
                # offset to center next point
                self.freqs = self.freqs + delta

#                print 'debug: center: %0.3f, , last center: %0.3f, freq_step: %0.3f, , delta: %0.3f, ' %\
#                        (center, self.last_center, self.freq_step, delta)
#                print 'debug; current center: %0.3f, current pt center: %0.3f, next pt center: %0.3f' %\
#                        (q_freqs_center, blah, np.average(self.freqs))
                self.last_center = center
                self.freq_step = delta


            # see if we should adjust spec power
            if width < self.width_min:
                self.amp *= 2
                # self.spec_power += 1
            elif width > self.width_max:
                self.amp /= 2
                # self.spec_power -= 1

            #evaluate if it's time to adjust the RO power
            if ((self.ro_pwr_tune != -1 and (idx % self.ro_pwr_tune) == 0)\
                or (self.ro_pwr_tune == -1 and np.absolute(self.last_center-self.last_tune_center) > self.ro_pwr_tune_interval))\
                and (self.ro_range != 0):

                self.qubit_rfsource.set_frequency(self.last_center)

                powers = np.arange(self.ro_power - self.ro_range,
                                     self.ro_power + self.ro_range + self.ro_step,
                                     self.ro_step)
#                r = ropcal(powers)
                r = orcpcal(powers, self.amp)
                r.measure()
                plt.close()

#                # update power
#                self.ro_power = r.get_best_power()
                self.ro_power = r.best_power

            self.analyze(idx)

        if self.source_type == 'VOLT':
            self.qubit_yoko.do_set_voltage(0)
        if self.source_type == 'CURR':
            self.qubit_yoko.do_set_current(0)
        self.qubit_yoko.set_output_state(0)

        # self.analyze()
        if self.savefig:
            self.save_fig()
        return self.set_vals, self.freqs, self.IQs[:]

    def analyze(self, index=0):
        fig = plt.figure(100)
        plt.clf()

        ax = fig.add_subplot(111)
        ax.plot(self.set_vals[0:index+1], self.center_freqs[0:index+1]/1e9, 'ks-')
        ax.set_title(self.data.get_fullname())
        ax.set_ylabel('Frequency (GHz)')
        if self.source_type == 'CURR':
            ax.set_xlabel('Current (A)')     
        elif self.source_type == 'VOLT':
            ax.set_xlabel('Voltage (V)') 

        fig = plt.figure(101)
        plt.clf()

        ax = fig.add_subplot(111)
        ymax = np.amax(self.freqs_log[0:index+1])
        ymin = np.amin(self.freqs_log[0:index+1])
        ystep = np.absolute(self.freqs[1]-self.freqs[0])
        ygrid = np.arange(ymin, ymax+ystep, ystep)
        # print [ymax, ymin, ystep]
        xgrid = self.set_vals

        data2D = np.zeros((self.num_steps, len(ygrid)))
        for ind, (curFreqs, curAmps) in enumerate(zip(self.freqs_log[0:index+1], self.amps[0:index+1])):
            freq_location = (np.absolute(np.subtract(ygrid, curFreqs[0]))).argmin()
            data2D[ind, freq_location:freq_location + len(curFreqs)] = curAmps

        plt.pcolor(xgrid, np.multiply(ygrid, 1e-9), np.transpose(data2D))
        plt.colorbar()
        ax.set_ylabel('Frequency (GHz)')
        if self.source_type == 'CURR':
            ax.set_xlabel('Current (A)')     
        elif self.source_type == 'VOLT':
            ax.set_xlabel('Voltage (V)')    

    def plotSpecs(self):
        fig = plt.figure(101)
        ax = fig.add_subplot(111)



    def get_spec_results(self):
        return self.set_vals[:], self.center_freqs[:], self.ro_log[:], self.spec_log[:]