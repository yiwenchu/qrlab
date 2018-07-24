from measurement import Measurement1D
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from lib.math import fitter
import time
import objectsharer as objsh

SPEC   = 0
POWER  = 1

class Tracked_Spectroscopy(Measurement1D):
    '''

    '''

    def __init__(self, qubit_info,
                 qubit_yoko,
                 voltages,
                 init_freqs, # program will handle updating frequency range
                 spec_params, # format (qubit_rfsource, init_spec_power)
                 init_ro_power,
                 ro_range = 7, #will search += xdbm
                 ro_step = 1,
                 ro_tune = 3, #tune every x experiments; -1 if no tuning
                 ro_shots = 1e4,
                 width_min=2e6, #Pulled these numbers out of rear.  plz fiz.
                 width_max=6e6,
                 freq_step=20e6,
                 plen=50e3, amp=1, seq=None, postseq=None,
                 pow_delay=1, freq_delay=0.1, plot_type=None,
                 use_IQge=0, use_weight=0,
                 subtraction=0,
                 **kwargs):

        self.qubit_info = qubit_info
        self.qubit_yoko = qubit_yoko

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
        self.ro_range = ro_range
        self.ro_step = ro_step

        self.ro_tune = ro_tune

        self.voltages = voltages

        self.use_IQge = use_IQge
        self.use_weight = use_weight
        self.subtraction = subtraction

        self.num_steps = len(self.voltages)

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
        self.yoko_voltages = self.data.create_dataset('yoko_voltages', shape=[self.num_steps])
        self.ro_log = self.data.create_dataset('ro_powers', shape=[self.num_steps])
        self.spec_log = self.data.create_dataset('spec_powers', shape=[self.num_steps])
        self.freqs_log = self.data.create_dataset('freqs', shape=[self.num_steps, self.num_freq_pts])
        self.center_freqs =  self.data.create_dataset('center_freqs', shape=[self.num_steps])
        self.widths =  self.data.create_dataset('widths', shape=[self.num_steps])

        self.IQs = self.data.create_dataset('avg', shape=[self.num_steps, self.num_freq_pts], dtype=np.complex)
        self.amps = self.data.create_dataset('avg_pp', shape=[self.num_steps, self.num_freq_pts])

    def measure(self):
        from scripts.single_qubit import spectroscopy
        from scripts.single_qubit import ropower_calibration
        from scripts.calibration import ro_power_cal

        spec = lambda q_freqs, ro_power, spec_power, yoko_voltage: \
            spectroscopy.Spectroscopy(
                                       self.qubit_info,
                                       q_freqs,
                                       [self.qubit_rfsource, spec_power],
                                       [ro_power],
                                       plen=self.plen,
                                       amp=self.amp,
                                       seq=self.seq,
                                       postseq=self.postseq,
                                       pow_delay=self.pow_delay,
                                       extra_info=self.extra_info,
                                       freq_delay=self.freq_delay,
                                       plot_type=self.plot_type,
                                       title='Pulsed spectroscopy: (ro, spec) = (%0.2f, %0.2f) dBm; yoko %0.3f V\n' % (ro_power, spec_power, yoko_voltage),
                                       analyze_data=True,
                                       keep_data=True,
                                       use_IQge=self.use_IQge,
                                       use_weight=self.use_weight,
                                       subtraction=self.subtraction)

        ropcal = lambda powers: ropower_calibration.ROPower_Calibration(
                self.qubit_info,
                powers,
                qubit_pulse=ropower_calibration.SAT_PULSE)

        orcpcal = lambda powers: ro_power_cal.Optimal_Readout_Power(
                self.qubit_info,
                powers,
                plen=50e3, update_readout=True, verbose_plots=False,
                plot_best=False,
                shots=self.ro_shots)

        # initialize yoko to a known voltage
        self.qubit_yoko.set_voltage(0)
        self.qubit_yoko.set_output_state(1)

        time.sleep(1)
        self.last_center = np.average(self.freqs) # middle point
        for idx, voltage in enumerate(self.voltages):
            self.qubit_yoko.set_voltage(voltage)

            # run spec at this voltage
            spec_exp = spec(self.freqs, self.ro_power, self.spec_power, voltage)
            spec_exp.measure()

            center = spec_exp.fit_params['x0'].value * 1e6
            width = spec_exp.fit_params['w'].value * 1e6

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
                if not ll < delta < ul:
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

            #save RO power, spec power, freqs, center, width, raw data..
            self.IQs[idx,:] = spec_exp.IQs[0,:]
            self.amps[idx,:] = spec_exp.amps[0,:]

            # save spec fit and experiment parameter data
            self.center_freqs[idx] = center
            self.widths[idx] = width
            self.yoko_voltages[idx] = voltage
            self.ro_log[idx] = self.ro_power
            self.spec_log[idx] = self.spec_power
            self.freqs_log[idx,:] = self.freqs

            # see if we should adjust spec power
            if width < self.width_min:
                self.spec_power += 1
            elif width > self.width_max:
                self.spec_power -= 1

            #evaluate if it's time to adjust the RO power
            if self.ro_tune != -1 and (idx % self.ro_tune) == 0:
                self.qubit_rfsource.set_frequency(self.last_center)

                powers = np.arange(self.ro_power - self.ro_range,
                                     self.ro_power + self.ro_range + 1,
                                     self.ro_step)
#                r = ropcal(powers)
                r = orcpcal(powers)
                r.measure()

#                # update power
#                self.ro_power = r.get_best_power()
                self.ro_power = r.best_power

        self.qubit_yoko.set_voltage(0)

        self.analyze()
        if self.savefig:
            self.save_fig()
        return self.voltages, self.freqs, self.IQs[:]

    def analyze(self):
        fig = plt.figure()

        ax = fig.add_subplot(111)
        ax.plot(self.voltages[:], self.center_freqs[:]/1e9, 'ks-')

        ax.set_title(self.data.get_fullname())

    def get_spec_results(self):
        return self.voltages[:], self.center_freqs[:], self.ro_log[:], self.spec_log[:]