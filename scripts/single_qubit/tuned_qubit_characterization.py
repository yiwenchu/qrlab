'''
Do various qubit characterization experiments while varying yoko current or voltage for flux tuning or field dependence measurements. 

Adapted from tracked_spectroscopy_yoko.py 

Author: Yiwen
Date: 05/28/2015
'''


from measurement import Measurement1D
import matplotlib.pyplot as plt
from pulseseq.sequencer import *
from pulseseq.pulselib import *
from lib.math import fitter
import time
import objectsharer as objsh
import numpy as np
import mclient
from mclient import instruments
from pulseseq import sequencer
from pulseseq import pulselib



SPEC   = 0
POWER  = 1

class Tuned_Qubit(Measurement1D):
    '''
    The types of experiments to do are specified in experiments. Each one is associated with determining and setting a certain parameter:
    ROFREQ: does cavity spectroscopy, sets readout frequency. 
    ROPWR: does readout calibration to determine optimal readout power and optimal dispersive readout parameters. This is performed after each run, and requires 
    For the experiments that should be performed, arguments specific to each experiment should be passed in. 
    Otherwise, if using a known functional form for determining values (such as readout frequency, qubit frequency, etc) in order to perform subsequent experiments (such as T1, etc.),
    pass in a function (eg an extrapolation function based on previous data)/
    If neither of these are done, nothing will be changed with respect to that type of experiment.

    '''

    def __init__(self, qubit_info, #pass in qubit info name as a string because we need both the info and the instrument to set things like pi amp
                 qubit_rfsource,
                 funcgen,
                 alazar,
                 source_type, #choices are CURR or VOLT for B field tuning, STARK for Stark shift with changing power, STARKSSB for Stark shift with changing SSB amplitude
                 set_vals, #current or voltage values for field tuning. Power for Stark
                 experiments, # which experiments to do in array of strings. Options so far are ROFREQ, ROPWR, SPEC, RABI, T1, T2, T2Echo(does CPMG with N=1)
                 qubit_yoko = None,
                 stark_rfsource = None,
                 stark_V_channel = 'ch3',
                 save_fig = True,

                 #readout frequency parameters (only used if doing ROFREQ experiments)                 
                 init_ro_freq = 9e9,
                 ro_freq_tune = -1, #tune freq every x experiments; -1 if using frequency interval only;
                 ro_freq_tune_interval = -1, #tune ro frequency if the qubit frequency has changed by this much since last tuning, -1 if using steps only
                 ro_spec_range = 10e6,

                 #pre-determined function for readout values (used if not doing RO calibration)
                 ro_freq_fxn = None,

                 # readout power parameters (only used if doing ROPWR experiments)
                 init_ro_power = 0,
                 ropwr_qubit_rf_pwr = 0,
                 ro_range = 2, #will search += xdbm, 0 if not doing power tuning 
                 ro_step = 1, #step to take in power
                 ro_pwr_tune = 1, #tune ro power every this many runs. -1 if using qubit frequency to determine when to tune
                 ro_pwr_tune_interval = 100e6, #tune ro pwr if the qubit frequency has changed by this much since last tuning, -1 if using steps only
                 ro_shots = 1e4,
                 ro_pwr_pi = False, 

                 #pre-determined function for readout values (used if not doing RO calibration)
                 ro_pwr_fxn = None,

 
                 #spec parameters (only used if doing SPEC or SSBSPEC experiments) 
                 init_freqs = [5e5], 
                 init_spec_power = 0,
                 spec_funcgen_freq = 5000,
                 spec_avgs = 3000,
                 width_min=2e6, #min and max width of qubit peak. Used to determine when to change spec power and when we've lost the qubit
                 width_max=6e6,
                 freq_step=0e6, # If 0, same frequencies will be used. Otherwise program will handle updating frequency range based on preview qubit frequencies
                 plen=50e3, amp=0.01, seq=None, simulseq = None, postseq=None, starkw = 100, starkch = 3, #starkw is the risetime of the gaussiansquare pulse 
                 pow_delay=1, freq_delay=0.1, plot_type=None,
                 use_IQge=0, use_weight=0,
                 subtraction=False,
                 spec_bgcor = False,
                 spec_generate = True,


                 #pre-determined function for qubit frequencies (used if not doing qubit spec)
                 qubit_freq_fxn = None,


                 #Rabi parameters (only used if doing RABI experiments)
                 rabi_rf_pwr = 0,
                 init_rabi_amps = [0],
                 rabi_funcgen_freq = 5000,
                 rabi_avgs = 1000,
                 rabi_pulse_len = 20,

                 #pre-determined function for pi pulse amplitude (used if not doing rabi)
                 pi_amp_fxn = None,


                 #T1 parameters (only used if doing T1 experiments)
                 T1_rf_pwr = 0,
                 T1_funcgen_freq = 1000,
                 T1_avgs = 1000,
                 init_T1_delays = [0],
                 T1_update_delays = True,
                 T1_bgcor = False,

                 #T2 parameters (only used if doing T2 experiments)
                 T2_rf_pwr = 0,
                 T2_funcgen_freq = 1000,
                 T2_delta = 300e3,
                 T2_avgs = 1000,
                 init_T2_delays = [0],
                 T2_set_f = False,                 

                 #T2Echo parameters (only used if doing T2Echo experiments)
                 T2E_rf_pwr = 0,
                 T2E_funcgen_freq = 1000,
                 T2E_delta = 0,
                 T2E_avgs = 1000,
                 init_T2E_delays = [0],

                 #Phonon T1 parameters
                 init_PhT1_delays = [0],
                 PhT1_funcgen_freq = 1000,
                 PhT1_avgs = 1000,

                 Ph_amp = 0,
                 Ph_piLength = 0,
                 Ph_sigma = 10,

                 **kwargs):

        self.qubit_info = mclient.get_qubit_info(qubit_info)
        self.qubit_inst = instruments[qubit_info]
        self.qubit_rfsource = qubit_rfsource
        self.qubit_yoko = qubit_yoko
        self.stark_rfsource = stark_rfsource
        self.stark_V_channel = stark_V_channel
        self.funcgen = funcgen
        self.alazar = alazar
        self.source_type = source_type
        self.experiments = experiments
        self.set_vals = set_vals
        self.num_steps = len(self.set_vals)
        self.save_fig = save_fig

        self.ro_shots = ro_shots
        self.ro_power = init_ro_power #for adjusting RO power
        self.ro_freq = init_ro_freq
        self.ro_range = ro_range
        self.ro_step = ro_step
        self.ro_pwr_tune = ro_pwr_tune
        self.ro_freq_tune = ro_freq_tune
        self.ro_freq_tune_interval = ro_freq_tune_interval
        self.ro_pwr_tune_interval = ro_pwr_tune_interval
        self.ropwr_qubit_rf_pwr = ropwr_qubit_rf_pwr
        self.ro_spec_range = ro_spec_range
        self.ro_freq_fxn = ro_freq_fxn
        self.ro_pwr_fxn = ro_pwr_fxn
        self.ro_pwr_pi = ro_pwr_pi

        
        self.freqs = init_freqs
        self.num_freq_pts = len(init_freqs)
        self.freq_range = max(init_freqs)-min(init_freqs)
        self.spec_power = init_spec_power
        self.spec_funcgen_freq = spec_funcgen_freq
        self.spec_avgs = spec_avgs
        self.width_min = width_min #for adjusting spec power
        self.width_max = width_max
        self.freq_step = freq_step # frequency step estimator,
                                   # Set to 0 if you want constant frequency
        self.plen = plen
        self.amp = amp
        self.pow_delay = pow_delay
        self.freq_delay = freq_delay
        if seq is None:
            seq = Trigger(250)
        self.seq = seq
        self.simulseq = simulseq
        self.postseq = postseq
        self.starkw = starkw
        self.starkch = starkch
        self.use_IQge = use_IQge
        self.use_weight = use_weight
        self.subtraction = subtraction
        self.spec_bgcor = spec_bgcor
        self.plot_type = plot_type
        self.extra_info = None
        self.qubit_freq_fxn = qubit_freq_fxn
        self.sideband_freq = self.qubit_info.deltaf
        self.spec_generate = spec_generate

        self.rabi_rf_pwr = rabi_rf_pwr
        self.init_rabi_amps = init_rabi_amps
        self.num_rabi_pts = len(init_rabi_amps)
        self.rabi_funcgen_freq = rabi_funcgen_freq
        self.rabi_avgs = rabi_avgs
        self.rabi_pulse_len = rabi_pulse_len
        self.pi_amp_fxn = pi_amp_fxn

        self.T1_rf_pwr = T1_rf_pwr
        self.T1_funcgen_freq = T1_funcgen_freq
        self.T1_avgs = T1_avgs
        self.T1_delays = init_T1_delays
        self.num_T1_pts = len(init_T1_delays)
        self.T1_update_delays = T1_update_delays
        self.T1_bgcor = T1_bgcor

        self.T2_rf_pwr = T2_rf_pwr
        self.T2_funcgen_freq = T2_funcgen_freq
        self.T2_delta = T2_delta
        self.T2_avgs = T2_avgs
        self.T2_delays = init_T2_delays
        self.T2_set_f = T2_set_f 
        self.num_T2_pts = len(init_T2_delays)              

        self.T2E_rf_pwr = T2E_rf_pwr
        self.T2E_funcgen_freq = T2E_funcgen_freq
        self.T2E_delta = T2E_delta
        self.T2E_avgs = T2E_avgs
        self.T2E_delays = init_T2E_delays
        self.num_T2E_pts = len(init_T2E_delays)

        self.PhT1_delays = init_PhT1_delays
        self.PhT1_funcgen_freq = PhT1_funcgen_freq
        self.PhT1_avgs = PhT1_avgs

        self.Ph_amp = Ph_amp
        self.Ph_piLength = Ph_piLength
        self.Ph_sigma = Ph_sigma
        self.num_PhT1_pts = len(init_PhT1_delays)



        super(Tuned_Qubit, self).__init__(1, infos=qubit_info, **kwargs)
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

        self.rabi_amps_log = self.data.create_dataset('rabi_amps', shape=[self.num_steps, self.num_rabi_pts])
        self.rabi_dat = self.data.create_dataset('rabi_dat', shape=[self.num_steps, self.num_rabi_pts])
        self.rabi_A =  self.data.create_dataset('rabi_A', shape=[self.num_steps])
        self.rabi_f =  self.data.create_dataset('rabi_f', shape=[self.num_steps])
        self.rabi_ph =  self.data.create_dataset('rabi_phase', shape=[self.num_steps])
        self.pi_amp =  self.data.create_dataset('pi_amp', shape=[self.num_steps])

        self.T1_delays_log = self.data.create_dataset('T1_delays', shape=[self.num_steps, self.num_T1_pts])
        self.T1_dat = self.data.create_dataset('T1_dat', shape=[self.num_steps, self.num_T1_pts])
        self.T1_A =  self.data.create_dataset('T1_A', shape=[self.num_steps])
        self.T1_A_err = self.data.create_dataset('T1_A_err', shape=[self.num_steps])
        self.T1_tau =  self.data.create_dataset('T1_tau', shape=[self.num_steps])
        self.T1_tau_err =  self.data.create_dataset('T1_tau_err', shape=[self.num_steps])

        self.T2_delays_log = self.data.create_dataset('T2_delays', shape=[self.num_steps, self.num_T2_pts])
        self.T2_dat = self.data.create_dataset('T2_dat', shape=[self.num_steps, self.num_T2_pts])
        self.T2_tau =  self.data.create_dataset('T2_tau', shape=[self.num_steps])
        self.T2_tau_err =  self.data.create_dataset('T2_tau_err', shape=[self.num_steps])
        self.T2_f =  self.data.create_dataset('T2_f', shape=[self.num_steps])
        self.T2_f_err =  self.data.create_dataset('T2_f_err', shape=[self.num_steps])

        self.T2E_delays_log = self.data.create_dataset('T2E_delays', shape=[self.num_steps, self.num_T2E_pts])
        self.T2E_dat = self.data.create_dataset('T2E_dat', shape=[self.num_steps, self.num_T2E_pts])
        self.T2E_tau =  self.data.create_dataset('T2E_tau', shape=[self.num_steps])
        self.T2E_tau_err =  self.data.create_dataset('T2E_tau_err', shape=[self.num_steps])
        self.T2E_A =  self.data.create_dataset('T2E_A', shape=[self.num_steps])
        self.T2E_A_err = self.data.create_dataset('T2E_A_err', shape=[self.num_steps])

        self.PhT1_delays_log = self.data.create_dataset('Ph_T1_delays', shape=[self.num_steps, self.num_PhT1_pts])
        self.PhT1_dat = self.data.create_dataset('Ph_T1_dat', shape=[self.num_steps, self.num_PhT1_pts])
        self.PhT1_A =  self.data.create_dataset('Ph_T1_A', shape=[self.num_steps])
        self.PhT1_A_err = self.data.create_dataset('Ph_T1_A_err', shape=[self.num_steps])
        self.PhT1_tau =  self.data.create_dataset('Ph_T1_tau', shape=[self.num_steps])
        self.PhT1_tau_err =  self.data.create_dataset('Ph_T1_tau_err', shape=[self.num_steps])

    def measure(self):
        from scripts.single_qubit import spectroscopy
        from scripts.calibration import ropower_calibration
        from scripts.calibration import ro_power_cal
        from scripts.single_cavity import rocavspectroscopy
        from scripts.single_qubit import rabi
        from scripts.single_qubit import T1measurement
        from scripts.single_qubit import T2measurement
        from scripts.single_qubit import ssbspec_fit, stark_swap


        spec = lambda q_freqs, ro_freq, ro_power, yoko_set_val: \
            spectroscopy.Spectroscopy(
                                       self.qubit_info,
                                       q_freqs,
                                       [self.qubit_rfsource, self.spec_power],
                                       # [ro_power],
                                       plen=self.plen,
                                       amp=self.amp,
                                       seq=self.seq,
                                       postseq=self.postseq,
                                       # pow_delay=self.pow_delay,
                                       extra_info=self.extra_info,
                                       freq_delay=self.freq_delay,
                                       plot_type=self.plot_type,
                                       title='Spec: (ro freq, ro pwr)=(%0.4e, %0.2f); setpt %0.3e V\n' % (ro_freq, ro_power, yoko_set_val),
                                       analyze_data=True,
                                       keep_data=True,
                                       use_IQge=self.use_IQge,
                                       use_weight=self.use_weight,
                                       subtraction=self.subtraction)

        SSBspec = lambda q_freqs, ro_freq, ro_power, yoko_set_val, b_freq, simulseq: \
            ssbspec_fit.SSBSpec_fit(
                                    self.qubit_info, q_freqs, 
                                    seq=self.seq, 
                                    simulseq = simulseq,
                                    postseq = self.postseq,
                                    bgcor = self.spec_bgcor,
                                    extra_info=self.extra_info,
                                    title='Spec: (ro freq, ro pwr)=(%0.4e, %0.2f); setpt %0.3e V\n brick_freq: %0.4e\n' % (ro_freq, ro_power, yoko_set_val, b_freq),
                                    generate = self.spec_generate,
                                    )

        ropcal = lambda powers: ropower_calibration.ROPower_Calibration(
                self.qubit_info,
                powers,
                qubit_pulse=ropower_calibration.SAT_PULSE)


        orcpcal = lambda powers, curAmp, curplen: ro_power_cal.Optimal_Readout_Power(
                self.qubit_info,
                powers,
                plen=curplen, amp=curAmp, update_readout=True, verbose_plots=False,
                plot_best=False,
                shots=self.ro_shots)

        rospec = lambda power, ro_freqs: rocavspectroscopy.ROCavSpectroscopy(self.qubit_info, [power], ro_freqs)

        rb = lambda amps, ro_freq, ro_power, yoko_set_val: rabi.Rabi(self.qubit_info, amps, r_axis = 0,
                       seq = None, postseq = None,
                       plot_seqs=False,
                       update=False,
                       title='Rabi: (ro freq, ro pwr)=(%0.4e, %0.2f); setpt %0.3e V\n' % (ro_freq, ro_power, yoko_set_val),
                       )

        T1 = lambda delays, ro_freq, ro_power, yoko_set_val: T1measurement.T1Measurement(self.qubit_info, delays, 
                                     double_exp=False,
                                     bgcor = self.T1_bgcor,
                                     title='T1: (ro freq, ro pwr)=(%0.4e, %0.2f); setpt %0.3e V\n' % (ro_freq, ro_power, yoko_set_val),
                                     )

        T2 = lambda delays, detune, ro_freq, ro_power, yoko_set_val: T2measurement.T2Measurement(self.qubit_info, delays, detune, 
                                     title='T2: (ro freq, ro pwr)=(%0.4e, %0.2f); setpt %0.3e V\n' % (ro_freq, ro_power, yoko_set_val),
                                     )

        T2Echo = lambda delays, detune, ro_freq, ro_power, yoko_set_val: T2measurement.T2Measurement(self.qubit_info, delays, detune, echotype = T2measurement.ECHO_CPMG,
                                     fit_type = 'gaussian_decay', necho = 1,
                                     title='T2CPMG(N=1): (ro freq, ro pwr)=(%0.4e, %0.2f); setpt %0.3e V\n' % (ro_freq, ro_power, yoko_set_val),
                                     )

        PhT1 = lambda delays, ro_freq, ro_power, set_val: stark_swap.phonon_T1(self.qubit_info, delays, 
                                     phonon_pi = self.Ph_piLength, amp = self.Ph_amp-set_val, sigma = self.Ph_sigma,
                                     title='T1: (ro freq, ro pwr)=(%0.4e, %0.2f); setpt %0.3e V\n' % (ro_freq, ro_power, set_val),
                                     )


        # initialize yoko to a known set_val
        if self.qubit_yoko != None:
            # if self.source_type == 'VOLT':
            #     self.qubit_yoko.do_set_voltage(0)
            # elif self.source_type == 'CURR':
            #     self.qubit_yoko.do_set_current(0)
            # else:
            #     print 'source type needs to be VOLT or CURR'
            #     return False
            self.qubit_yoko.set_output_state(1)

        # if self.stark_rfsource != None:
            # self.stark_rfsource.set_power(-40)
            # self.stark_rfsource.set_rf_on(1)

        time.sleep(1)
        if 'SPEC' in self.experiments:
            self.last_center = np.average(self.freqs) # middle point
        elif 'SSBSPEC' in self.experiments:
            self.last_center = self.qubit_rfsource.get_frequency()+self.sideband_freq # use current setting of brick
            self.next_center = self.qubit_rfsource.get_frequency()+self.sideband_freq
        elif self.qubit_freq_fxn!=None:
            self.last_center = self.qubit_freq_fxn(self.set_vals[0])
        else:
            self.last_center = 0 #Not doing spec or changing frequency, so keeping same qubit frequency. Presumably this means we're not tuning pwr either, so this case should never be used.

        self.last_freq_tune_center = self.last_center
        self.last_pwr_tune_center = self.last_center
        self.ro_source.set_frequency(self.ro_freq)
        self.ro_source.set_power(self.ro_power)

        for idx, set_val in enumerate(self.set_vals):
            if self.source_type == 'VOLT':
                self.qubit_yoko.do_set_voltage(float(set_val))
            elif self.source_type == 'CURR':
                self.qubit_yoko.do_set_current(float(set_val))
            elif self.source_type == 'STARK':
                self.stark_rfsource.set_power(float(set_val))
            elif self.source_type == 'STARK_V':
                if self.stark_V_channel == 'ch3':
                    self.stark_rfsource.set_ch3_offset(float(set_val))
            elif self.source_type == 'STARKSSB':
                self.simulseq = sequencer.Sequence(pulselib.GaussSquare(self.qubit_info.w_selective*4+2*self.starkw, float(set_val), self.starkw, chan = self.starkch))


            time.sleep(1)
            self.yoko_set_vals[idx] = set_val

            # ro frequency
            if 'ROFREQ' in self.experiments:
                if ((self.ro_freq_tune != -1 and (idx % self.ro_freq_tune) == 0)\
                    or (self.ro_freq_tune == -1 and np.absolute(self.last_center-self.last_freq_tune_center) > self.ro_freq_tune_interval))\
                    and (self.ro_spec_range !=0): #tune if it's been ro_freq_tune runs or, if not, the qubit has shifted by ro_freq_tune_interval

                    self.funcgen.set_frequency(5000)  #rep rate    
                    self.alazar.set_naverages(3000)
                    self.qubit_rfsource.set_rf_on(0) #in case rf source is resonant with qubit. This would mess up readout spec

                    ro_freqs = np.linspace(self.ro_freq - self.ro_spec_range, 
                                            self.ro_freq + self.ro_spec_range, 
                                            51)
                    ro = rospec(self.ro_power, ro_freqs)
                    ro.measure()
                    plt.close()

                    self.qubit_rfsource.set_rf_on(1)

                    self.ro_freq = ro.fit_params[0][2]

                    self.last_freq_tune_center = self.last_center

            elif self.ro_freq_fxn!=None:
                self.ro_freq = self.ro_freq_fxn(set_val)
                

            self.ro_source.set_frequency(self.ro_freq)
            self.ro_LO.set_frequency(self.ro_freq+50e6)
            self.ro_freq_log[idx] = self.ro_freq
            self.ro_pwr_log[idx] = self.ro_power



            # qubit saturation spec
            if 'SPEC' in self.experiments: 

                self.funcgen.set_frequency(self.spec_funcgen_freq)  #rep rate    
                self.alazar.set_naverages(self.spec_avgs)

                spec_exp = spec(self.freqs, self.ro_freq, self.ro_power, set_val)
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
                self.widths[idx] = width
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

            elif self.qubit_freq_fxn!=None:
                self.last_center = self.qubit_freq_fxn(set_val)

            self.qubit_rfsource.set_frequency(self.last_center)
            self.center_freqs[idx] = self.last_center

            # qubit SSB spec
            if 'SSBSPEC' in self.experiments: 

                # self.qubit_inst.set_w(self.plen)
                # self.qubit_inst.set_pi_amp(self.amp)

                self.funcgen.set_frequency(self.spec_funcgen_freq)  #rep rate    
                self.alazar.set_naverages(self.spec_avgs)


                self.qubit_rfsource.set_frequency(self.next_center-self.sideband_freq)
                spec_exp = SSBspec(self.freqs, self.ro_freq, self.ro_power, set_val, self.qubit_rfsource.get_frequency(), self.simulseq)
                spec_exp.measure()
                plt.close()

                center = self.next_center + np.sign(self.sideband_freq)*spec_exp.fit_params['x0'].value
                width = spec_exp.fit_params['w'].value

                #save RO power, spec power, freqs, center, width, raw data..
                # self.IQs[idx,:] = spec_exp.IQs[0,:]
                # self.amps[idx,:] = spec_exp.amps[0,:]

                # self.IQs[idx,:] = spec_exp.IQs[:]
                self.amps[idx,:] = spec_exp.get_ys()  
                # save spec fit and experiment parameter data
                self.widths[idx] = width
                self.spec_log[idx] = self.spec_power
                if np.sign(self.sideband_freq)<0:
                    self.freqs_log[idx,:] = self.next_center - np.flipud(self.freqs)
                    self.amps[idx,:] = np.flipud(spec_exp.get_ys())
                else:
                    self.freqs_log[idx,:] = self.next_center + self.freqs
                    self.amps[idx,:] = spec_exp.get_ys()                   

                

                #choose the next frequency range
                delta = center - self.last_center
                self.last_center = center

                if self.freq_step == 0:
                    # no step, keep the same center frequency
                    self.next_center = self.next_center
                elif idx == 0: # first point; don't try to stop
                    self.next_center = center + self.freq_step
                else:
                    # use 0th order estimator from last points
                    # we will weight the frequency step to the expected direction
                    # of the frequency step
                    # if self.freq_step > 0:
                    #     # increasing freq
                    #     ll, ul = np.array([-0.25, 1.25]) * self.freq_step
                    # else:
                    #     # decreasing freq
                    #     ll, ul = np.array([1.25, -0.25]) * self.freq_step

                    #Commented out 9/3/14 Jacob, not quite right.
                    # if not ll < delta < ul:
                    #     print 'quitting: we seem to have lost the qubit'
                    #     break

                    #if the width is off, probably a sign that we're not fitting to qubit peak
                    # if not self.width_min*0.5 < width < self.width_max*2:
                    #     print 'quitting: we seem to have lost the qubit'
                    #     break

                    # pick new frequency range such that the NEXT point should be centered
                    self.next_center = center + delta

    #                print 'debug: center: %0.3f, , last center: %0.3f, freq_step: %0.3f, , delta: %0.3f, ' %\
    #                        (center, self.last_center, self.freq_step, delta)
    #                print 'debug; current center: %0.3f, current pt center: %0.3f, next pt center: %0.3f' %\
    #                        (q_freqs_center, blah, np.average(self.freqs))
                    self.freq_step = delta


                # # see if we should adjust spec power
                # if width < self.width_min:
                #     self.amp *= 2
                #     # self.spec_power += 1
                # elif width > self.width_max:
                #     self.amp /= 2
                #     # self.spec_power -= 1

            elif self.qubit_freq_fxn!=None:
                self.last_center = self.qubit_freq_fxn(set_val)

            self.qubit_rfsource.set_frequency(self.last_center-self.sideband_freq)
            self.center_freqs[idx] = self.last_center

            if 'RABI' in self.experiments:
                
                # self.qubit_inst.set_w(self.rabi_pulse_len)
                self.funcgen.set_frequency(self.rabi_funcgen_freq)  #rep rate    
                self.alazar.set_naverages(self.rabi_avgs)
                self.qubit_rfsource.set_power(self.rabi_rf_pwr)

                rabi_exp = rb(self.init_rabi_amps, self.ro_freq, self.ro_power, set_val)
                rabi_exp.measure()
                plt.close()

                rabi_A = rabi_exp.fit_params['A'].value
                rabi_f = rabi_exp.fit_params['f'].value
                rabi_ph = rabi_exp.fit_params['dphi'].value

                #find pi pulse amplitude. Accounts for offsets
                pi_amp = (np.pi-np.pi/2*np.sign(rabi_A)-rabi_ph%(2*np.pi))/(2*np.pi*rabi_f)

                self.rabi_amps_log[idx,:] = rabi_exp.xs[:]
                self.rabi_dat[idx,:] = rabi_exp.get_ys()[:]      
                # save spec fit and experiment parameter data
                self.rabi_A[idx] = rabi_A
                self.rabi_f[idx] = rabi_f
                self.rabi_ph[idx] = rabi_ph

            elif self.pi_amp_fxn!=None:
                pi_amp = self.pi_amp_fxn(set_val)
            else: 
                pi_amp = 0

            self.pi_amp[idx] = pi_amp
            if np.absolute(pi_amp)>1e-3 and np.absolute(pi_amp)<1:
                self.qubit_inst.set_pi_amp(pi_amp)  


            if 'T1' in self.experiments:
                self.funcgen.set_frequency(self.T1_funcgen_freq)  #rep rate    
                self.alazar.set_naverages(self.T1_avgs)
                self.qubit_rfsource.set_power(self.T1_rf_pwr)

                T1_exp = T1(self.T1_delays, self.ro_freq, self.ro_power, set_val)
                T1_exp.measure()
                plt.close()

                T1_A = T1_exp.fit_params['A'].value
                T1_A_err = T1_exp.fit_params['A'].stderr
                T1_tau = T1_exp.fit_params['tau'].value
                T1_tau_err = T1_exp.fit_params['tau'].stderr

                self.T1_delays_log[idx,:] = T1_exp.xs[:]
                self.T1_dat[idx,:] = T1_exp.get_ys()[:]      
                # save spec fit and experiment parameter data
                self.T1_A[idx] = T1_A
                self.T1_A_err[idx] = T1_A_err
                self.T1_tau[idx] = T1_tau
                self.T1_tau_err[idx] = T1_tau_err
                
                #decide next T1 delays
                if self.T1_update_delays and T1_tau>0.001 and T1_tau<1000: #in case something goes wrong in the fitting 
                    self.T1_delays = np.linspace(0, T1_tau*1e3*8, self.num_T1_pts)


            if 'T2' in self.experiments:
                self.funcgen.set_frequency(self.T2_funcgen_freq)  #rep rate    
                self.alazar.set_naverages(self.T2_avgs)
                self.qubit_rfsource.set_power(self.T2_rf_pwr)

                T2_exp = T2(self.T2_delays, self.T2_delta, self.ro_freq, self.ro_power, set_val)
                T2_exp.measure()
                plt.close()

                T2_tau = T2_exp.fit_params['tau'].value
                T2_tau_err = T2_exp.fit_params['tau'].stderr
                T2_f = T2_exp.fit_params['f'].value
                T2_f_err = T2_exp.fit_params['f'].stderr

                self.T2_delays_log[idx,:] = T2_exp.xs[:]
                self.T2_dat[idx,:] = T2_exp.get_ys()[:]      
                # save spec fit and experiment parameter data
                self.T2_f[idx] = T2_f
                self.T2_f_err[idx] = T2_f_err
                self.T2_tau[idx] = T2_tau
                self.T2_tau_err[idx] = T2_tau_err
                
                if self.T2_set_f:
                    freq_offset = np.absolute(T2_f*1e6-self.T2_delta)
                    freq_old = self.last_center
                    ctr = 0
                    while freq_offset > 2e6/(2*np.pi*T1_tau):
                        ctr += 1
                        if ctr > 3:
                            print 'quitting: cannot find qubit frequency from Ramsey'
                            break 
                        else:
                            freq_cur = freq_old + freq_offset
                            self.qubit_rfsource.set_frequency(freq_cur)
                            T2_exp = T2(self.T2_delays, self.T2_delta, self.ro_freq, self.ro_power, set_val)
                            T2_exp.measure()
                            plt.close()                       
                            freq_offset_old = freq_offset
                            freq_offset = np.absolute(T2_exp.fit_params['f'].value*1e6-self.T2_delta)
                            if freq_offset <= freq_offset_old:
                                freq_old = freq_cur
                            if freq_offset > freq_offset_old:
                                freq_cur = freq_old - freq_offset_old
                                self.qubit_rfsource.set_frequency(freq_cur)
                                T2_exp = T2(self.T2_delays, self.T2_delta, self.ro_freq, self.ro_power, set_val)
                                T2_exp.measure()
                                plt.close()
                                freq_offset = np.absolute(T2_exp.fit_params['f'].value*1e6-self.T2_delta)
                                freq_old = freq_cur


                # set zero detuning based on Ramsey frequency
                # if self.T2_set_f:
                #     freq_offset = np.absolute(T2_f-self.T2_delta/1e6)
                #     if freq_offset > 1/(2*np.pi*T1_tau):
                #         self.qubit_rfsource.set_frequency(self.last_center + freq_offset*1e6)
                #         T2_exp = T2(self.T2_delays, self.T2_delta, self.ro_freq, self.ro_power, set_val)
                #         T2_exp.measure()
                #         plt.close()
                #         T2_f_new = T2_exp.fit_params['f'].value
                #     if np.absolute(T2_f_new-self.T2_delta/1e6) > 1/(2*np.pi*T1_tau):
                #         self.qubit_rfsource.set_frequency(self.last_center - freq_offset*1e6)
                #         T2_exp = T2(self.T2_delays, self.T2_delta, self.ro_freq, self.ro_power, set_val)
                #         T2_exp.measure()
                #         plt.close()
                #         T2_f_new = T2_exp.fit_params['f'].value
                #     if np.absolute(T2_f_new-self.T2_delta/1e6) > 1/(2*np.pi*T1_tau):
                #         print 'quitting: cannot find qubit frequency from Ramsey'
                #         break  

                #decide next T2 delays and detunings
                if T2_tau>0.001 and T2_tau<1000: #in case something goes wrong in the fitting 
                    self.T2_delays = np.linspace(0, T2_tau*1e3*3, self.num_T2_pts)
                    self.T2_delta = 3/T2_tau*1e6



            if 'T2Echo' in self.experiments:
                self.funcgen.set_frequency(self.T2E_funcgen_freq)  #rep rate    
                self.alazar.set_naverages(self.T2E_avgs)
                self.qubit_rfsource.set_power(self.T2E_rf_pwr)

                T2E_exp = T2Echo(self.T2E_delays, self.T2E_delta, self.ro_freq, self.ro_power, set_val)
                T2E_exp.measure()
                plt.close()

                T2E_tau = T2E_exp.fit_params['sigma'].value
                T2E_tau_err = T2E_exp.fit_params['sigma'].stderr
                T2E_A = T2E_exp.fit_params['area'].value / (2 * T2E_tau) / np.sqrt(np.pi / 2)
                T2E_A_err = np.absolute(T2E_A)*np.sqrt((T2E_tau_err/T2E_tau)**2+(T2E_exp.fit_params['area'].stderr/T2E_A)**2)


                self.T2E_delays_log[idx,:] = T2E_exp.xs[:]
                self.T2E_dat[idx,:] = T2E_exp.get_ys()[:]      
                # save spec fit and experiment parameter data
                self.T2E_A[idx] = T2E_A
                self.T2E_A_err[idx] = T2E_A_err
                self.T2E_tau[idx] = T2E_tau
                self.T2E_tau_err[idx] = T2E_tau_err

                if T2E_tau>0.001 and T2E_tau<1000: #in case something goes wrong in the fitting 
                    self.T2E_delays = np.linspace(0, T2E_tau*1e3*5, self.num_T2E_pts)

            if 'PhT1' in self.experiments:
                self.funcgen.set_frequency(self.PhT1_funcgen_freq)  #rep rate    
                self.alazar.set_naverages(self.PhT1_avgs)

                PhT1_exp = PhT1(self.PhT1_delays, self.ro_freq, self.ro_power, set_val)
                PhT1_exp.measure()
                plt.close()

                PhT1_A = PhT1_exp.fit_params['A'].value
                PhT1_A_err = PhT1_exp.fit_params['A'].stderr
                PhT1_tau = PhT1_exp.fit_params['tau'].value
                PhT1_tau_err = PhT1_exp.fit_params['tau'].stderr

                self.PhT1_delays_log[idx,:] = PhT1_exp.xs[:]
                self.PhT1_dat[idx,:] = PhT1_exp.get_ys()[:]      
                # save spec fit and experiment parameter data
                self.PhT1_A[idx] = PhT1_A
                self.PhT1_A_err[idx] = PhT1_A_err
                self.PhT1_tau[idx] = PhT1_tau
                self.PhT1_tau_err[idx] = PhT1_tau_err
                

            #RO power
            if 'ROPWR' in self.experiments:
                if ((self.ro_pwr_tune != -1 and (idx % self.ro_pwr_tune) == 0)\
                    or (self.ro_pwr_tune == -1 and np.absolute(self.last_center-self.last_pwr_tune_center) > self.ro_pwr_tune_interval)): #tune if it's been ro_pwr_tune runs or, if not, the qubit has shifted by ro_pwr_tune_interval

                    self.qubit_rfsource.set_frequency(self.last_center)
                    self.qubit_rfsource.set_power(self.ropwr_qubit_rf_pwr)

                    self.funcgen.set_frequency(5000)  #rep rate    
                    self.alazar.set_naverages(3000)

                    powers = np.arange(self.ro_power - self.ro_range,
                                         self.ro_power + self.ro_range + self.ro_step,
                                         self.ro_step)
    #                r = ropcal(powers)
                    if self.ro_pwr_pi == False:
                        r = orcpcal(powers, self.amp, self.plen)
                    else:
                        r = orcpcal(powers, self.amp, None)
                    r.measure()
                    plt.close()

    #                # update power
    #                self.ro_power = r.get_best_power()
                    self.ro_power = r.best_power

                    self.last_pwr_tune_center = self.last_center
            elif self.ro_pwr_fxn!=None:
                self.ro_power = self.ro_pwr_fxn(set_val)
                self.ro_source.set_power(self.ro_power)



            self.analyze(idx)

        # if self.qubit_yoko != None:
        #     if self.source_type == 'VOLT':
        #         self.qubit_yoko.do_set_voltage(0)
        #     elif self.source_type == 'CURR':
        #         self.qubit_yoko.do_set_current(0)

        #     self.qubit_yoko.set_output_state(0)


        if self.stark_rfsource != None:
            self.stark_rfsource.set_rf_on(0)

        # self.analyze()
        # if self.savefig:
        #     self.save_fig()
        return self.set_vals, self.freqs, self.IQs[:]

    def analyze(self, index=0):
        if ('SPEC' in self.experiments) or ('SSBSPEC' in self.experiments):
            fig = plt.figure(300)
            plt.clf()

            ax = fig.add_subplot(111)
            ax.plot(self.set_vals[0:index+1], self.center_freqs[0:index+1]/1e9, 'ks-')
            ax.set_title(self.data.get_fullname() + ' Qubit Frequency')
            ax.set_ylabel('Frequency (GHz)')
            if self.source_type == 'CURR':
                ax.set_xlabel('Current (A)')     
            elif self.source_type == 'VOLT':
                ax.set_xlabel('Voltage (V)') 

            fig = plt.figure(301)
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
                # print [freq_location, len(curFreqs)]
                data2D[ind, freq_location:freq_location + len(curFreqs)] = curAmps

            plt.pcolormesh(xgrid, np.multiply(ygrid, 1e-9), np.transpose(data2D))
            plt.colorbar()
            ax.set_title(self.data.get_fullname() + ' Spec')
            ax.set_ylabel('Frequency (GHz)')
            if self.source_type == 'CURR':
                ax.set_xlabel('Current (A)')     
            elif self.source_type == 'VOLT':
                ax.set_xlabel('Voltage (V)')  


        if 'RABI' in self.experiments:
            fig = plt.figure(302)
            plt.clf()

            ax = fig.add_subplot(111)            
            ax.plot(self.set_vals[0:index+1], self.pi_amp[0:index+1], 'ks-')
            ax.set_title(self.data.get_fullname() + ' pi Amplitude')
            ax.set_ylabel('Amplitude')
            if self.source_type == 'CURR':
                ax.set_xlabel('Current (A)')     
            elif self.source_type == 'VOLT':
                ax.set_xlabel('Voltage (V)')
            plt.show()


        if 'T1' in self.experiments:
            fig = plt.figure(303)
            plt.clf()

            ax = fig.add_subplot(111)            
            ax.errorbar(self.set_vals[0:index+1], self.T1_tau[0:index+1], self.T1_tau_err[0:index+1], fmt='go-')
            ax.set_title(self.data.get_fullname() + ' T1s')
            ax.set_ylabel('T1 (us)')
            if self.source_type == 'CURR':
                ax.set_xlabel('Current (A)')     
            elif self.source_type == 'VOLT':
                ax.set_xlabel('Voltage (V)') 
            plt.show()


            fig = plt.figure(307)
            plt.clf()

            ax = fig.add_subplot(111)
            ygrid = self.T1_delays
            # print [ymax, ymin, ystep]
            xgrid = self.set_vals

            data2D = np.zeros((self.num_steps, len(ygrid)))
            for ind, (curFreqs, curAmps) in enumerate(zip(self.T1_delays_log[0:index+1], self.T1_dat[0:index+1])):
                freq_location = (np.absolute(np.subtract(ygrid, curFreqs[0]))).argmin()
                # print [freq_location, len(curFreqs)]
                data2D[ind, freq_location:freq_location + len(curFreqs)] = curAmps

            plt.pcolormesh(xgrid/1e3, np.multiply(ygrid, 1e-9), np.transpose(data2D))
            plt.colorbar()
            ax.set_title(self.data.get_fullname() + ' T1s')
            ax.set_ylabel('Delay ($\mu$s)')
            if self.source_type == 'CURR':
                ax.set_xlabel('Current (A)')     
            elif self.source_type == 'VOLT':
                ax.set_xlabel('Voltage (V)')  


        if 'T2' in self.experiments:
            fig = plt.figure(304)
            plt.clf()

            ax = fig.add_subplot(111)            
            ax.errorbar(self.set_vals[0:index+1], self.T2_tau[0:index+1], self.T2_tau_err[0:index+1], fmt='go-')
            ax.set_title(self.data.get_fullname() + ' T2s')
            ax.set_ylabel('T2 (us)')
            if self.source_type == 'CURR':
                ax.set_xlabel('Current (A)')     
            elif self.source_type == 'VOLT':
                ax.set_xlabel('Voltage (V)') 
            plt.show()



        if 'T2Echo' in self.experiments:
            fig = plt.figure(305)
            plt.clf()

            ax = fig.add_subplot(111)            
            ax.errorbar(self.set_vals[0:index+1], self.T2E_tau[0:index+1], self.T2E_tau_err[0:index+1], fmt='go-')
            ax.set_title(self.data.get_fullname() + ' T2 CPMG')
            ax.set_ylabel('T2 (us)')
            if self.source_type == 'CURR':
                ax.set_xlabel('Current (A)')     
            elif self.source_type == 'VOLT':
                ax.set_xlabel('Voltage (V)') 
            plt.show()

        if 'PhT1' in self.experiments:
            fig = plt.figure(306)
            plt.clf()

            ax = fig.add_subplot(111)            
            ax.errorbar(self.set_vals[0:index+1], self.PhT1_tau[0:index+1], self.PhT1_tau_err[0:index+1], fmt='go-')
            ax.set_title(self.data.get_fullname() + ' Phonon T1s')
            ax.set_ylabel('PhT1 (us)')
            if self.source_type == 'CURR':
                ax.set_xlabel('Current (A)')     
            elif self.source_type == 'VOLT':
                ax.set_xlabel('Voltage (V)') 
            plt.show()

    def get_spec_results(self):
        return self.set_vals[:], self.center_freqs[:], self.ro_log[:], self.spec_log[:]