from lib.math.minimizer import Parameter, Minimizer
from mclient import instruments
import time
import numpy as np
import pulseseq.sequencer as sequencer
import pulseseq.pulselib as pulselib
import logging
import awgloader
import mclient
import config
import copy

def define_object(obj):
    '''Lets you pass objects either by their name or directly as mclient containers'''
    if type(obj) == str:
        return instruments[obj]
    else:
        return obj

#I believe thse can't be methods for lo_leakage_func to work on them directly.
#They could probably be absorbed with a lambda function, though.
def lo_leakage_func(params, awg, chans, spec, n_avg=1,delay=0.05, verbose=False):
    awg.set({
        'ch%s_offset'%chans[0]: params['vI'].value,
        'ch%s_offset'%chans[1]: params['vQ'].value,
    })
    time.sleep(delay)
    val = np.average([spec.get_power() for _ in range(n_avg)])
    if verbose:
        print 'Measuring at (%.06f, %.06f): %.01f' % \
                (params['vI'].value, params['vQ'].value, val)
    return val

def osb_min_func(params, awg, chans, spec, n_avg=1, delay=0.05, chan_select='q', verbose=False):
    if chan_select == 'i':
        ch_tune, ch_base = chans
    elif chan_select == 'q':
        ch_base, ch_tune = chans
    ch_amp = awg.get('ch%s_amplitude'%ch_base)
    awg.set({
        'ch%s_skew'%ch_tune: params['skew'].value,
        'ch%s_amplitude'%ch_tune: (params['amplitude'].value * ch_amp)
    })
    time.sleep(delay)
    val = np.average([spec.get_power() for _ in range(n_avg)])
    if verbose:
        print 'Measuring at (%.06f, %.06f): %.01f' % \
                (params['skew'].value, params['amplitude'].value, val)

    return val


def lo_leakage_func_SH(params, awg, chans, spec, freq, n_avg=1,delay=0.05, 
                       verbose=False):
    awg.set({
        'ch%s_offset'%chans[0]: params['vI'].value,
        'ch%s_offset'%chans[1]: params['vQ'].value,
    })
    time.sleep(delay)
    val = spec.read_single_freq(freq, n_av=n_avg)
    
    if verbose:
        print 'Measuring at (%.06f, %.06f): %.01f' % \
                (params['vI'].value, params['vQ'].value, val)
    return val

def osb_min_func_SH(params, awg, chans, spec, freq, n_avg=1, delay=0.05, chan_select='q', verbose=False):
    if chan_select == 'i':
        ch_tune, ch_base = chans
    elif chan_select == 'q':
        ch_base, ch_tune = chans
    ch_amp = awg.get('ch%s_amplitude'%ch_base)
    awg.set({
        'ch%s_skew'%ch_tune: params['skew'].value,
        'ch%s_amplitude'%ch_tune: (params['amplitude'].value * ch_amp)
    })
    time.sleep(delay)
    val = spec.read_single_freq(freq, n_av=n_avg)
    
    if verbose:
        print 'Measuring at (%.06f, %.06f): %.01f' % \
                (params['skew'].value, params['amplitude'].value, val)

    return val


class Mixer_Calibration(object):
    '''
    Example code:

    cal = mixer_cal('ancilla', 7759.634e6, 'vspec', 'AWG2', verbose=True,
                        base_amplitude=2.0,
                        va_lo='va_lo_low')

    cal.prep_instruments(reset_offsets=True, reset_ampskew=True)
    cal.tune_lo(mode='coarse')
    cal.tune_osb(mode=(0.2, 2000, 3, 1))
    cal.tune_lo(mode='fine') # useful if using 10 dB attenuation;
                            # LO leakage may creep up during osb tuning

    # this function will set the correct qubit_info sideband phase for use in experiments
    #    i.e. combines the AWG skew with the current sideband phase offset
    cal.set_tuning_parameters(set_sideband_phase=True)
    cal.load_test_waveform()
    cal.print_tuning_parameters()
    '''

    def __init__(self, qubit_info, frequency, spec='vspec',
                 va_lo=None, verbose=False,
                 base_amplitude=1, osb_chan='q'):
        '''
        Frequency here should be the actual qubit frequency.  We'll find
        whether or not you want to use the upper or lower sideband from the
        qubit info's modulation parameters

        The spec object contains internally the correct if frequency for the
        spectrum analyzer, so the end user does not need to know this once its
        set the first time.
        '''
        self.qubit_info = define_object(qubit_info)

        self.spec = define_object(spec)
        if self.spec.get_type() == 'SA124B':
            self.use_SH = True
        elif self.spec.get_type() == 'Vlastakis_Spec':
            self.use_SH = False
        else:
            raise ValueError('spec instruments not recognized!')
            
        self.va_lo = va_lo

        self.verbose = verbose
        self.base_amplitude = base_amplitude

        self.chans = [int(x) for x in self.qubit_info.get_channels().split(',')]
        self.full_chans = copy.copy(self.chans)

        awg_num = (self.chans[0]-1)//4+1
        self.awg = define_object('AWG%d' % (awg_num,))

        self.chans = [(ch-1)%4+1 for ch in self.chans]
        self.osb_chan = osb_chan

        self.if_freq = self.qubit_info.get_deltaf()

        if self.if_freq > 0: # positive IF = USB
            self.shift_phase = 0
        else: # negative IF = LSB
            self.shift_phase = 0

        self.lo_frequency = frequency - self.if_freq
        self.osb_freq = frequency - 2*self.if_freq
        self.set_min_funcs()

    def set_min_funcs(self):
        if self.use_SH:
            self.lo_leakage_func = lo_leakage_func_SH
            self.osb_min_func = osb_min_func_SH
        else:
            self.lo_leakage_func = lo_leakage_func
            self.osb_min_func = osb_min_func

    def prep_instruments(self, reset_offsets=False, reset_ampskew=False):
        '''
            1. Prepares AWG for modulation - select <reset_offsets> and/or <reset_ampskew>
                to reset offsets and/or amplitude imbalance/skew settings
            2. Loads test waveform
            3. Prepares IQ LO and vspec settings.
                Does NOT set the frequency for the VALO
        '''
        print "prepping instruments"

        self.awg.delete_all_waveforms()
        self.awg.get_id() #Sometimes AWGs are slow.  If it responds it's ready.

        if reset_offsets:
            self.awg.set({
                'ch%s_offset' % self.chans[0]: 0,
                'ch%s_offset' % self.chans[1]: 0
                })
        if reset_ampskew:
            self.qubit_info.set_sideband_phase(0)
            self.awg.set({
                'ch%s_amplitude'%self.chans[0]: self.base_amplitude,
                'ch%s_amplitude'%self.chans[1]: self.base_amplitude,
                'ch%s_skew'%self.chans[0]: 0,
                'ch%s_skew'%self.chans[1]: 0})

        self.get_tuning_parameters()
        self.load_test_waveform()

        if not self.use_SH:
            self.old_valo = self.spec.get_rfsource()
            if self.va_lo is not None:
                self.spec.set_rfsource(self.va_lo)
                print 'setting VALO: %s'% (self.va_lo)
    
            valo_inst = instruments[self.va_lo]
            valo_inst.set_power(10)
            valo_inst.set_use_extref(True)
    #        valo_inst.set_use_ext_pulse_mod(False)
            time.sleep(1)

    def reset_valo(self):
        if not self.use_SH:
            self.spec.set_rfsource(self.old_valo)

    def get_tuning_parameters(self):
        '''
            update current settings as optimal settings
        '''
        self.opt_offset_I = self.awg.get('ch%s_offset' % self.chans[0])
        self.opt_offset_Q = self.awg.get('ch%s_offset' % self.chans[1])

        self.chI_amplitude = self.awg.get('ch%s_amplitude' % self.chans[0])
        self.chQ_amplitude = self.awg.get('ch%s_amplitude' % self.chans[1])

        self.opt_skew_ps = self.awg.get('ch%s_skew' % self.chans[1])
        self.opt_amp_factor = self.chQ_amplitude / self.chI_amplitude

        self.sideband_phase = self.qubit_info.get_sideband_phase()

    def set_tuning_parameters(self, set_sideband_phase=False):
        '''
            Set current optimal settings.
            If <set_sideband_phase>, then convert AWG skew delay to qubit_info
            sideband_phase.
        '''
        self.awg.set({
                'ch%s_offset' % self.chans[0]: self.opt_offset_I,
                'ch%s_offset' % self.chans[1]: self.opt_offset_Q,
                'ch%s_amplitude'%self.chans[0]: self.base_amplitude,
                'ch%s_amplitude'%self.chans[1]: self.base_amplitude * self.opt_amp_factor,
                'ch%s_skew'%self.chans[0]: 0,
                'ch%s_skew'%self.chans[1]: 0})

        if set_sideband_phase:
            # use this for final tuneup for SSB
            sideband_phase =  self.calculate_sideband_phase(self.opt_skew_ps,
                                                            total_phase=True)
            self.qubit_info.set_sideband_phase(sideband_phase)
        else:
            self.qubit_info.set_sideband_phase(self.sideband_phase)
            self.awg.set({'ch%s_skew'%self.chans[1]: self.opt_skew_ps})

    def print_tuning_parameters(self):
        print '-------- SSB Tuning parameters: %s -------' % (self.qubit_info.get_name())
        print 'I, Q offsets = (%0.3f, %0.3f)' % (self.opt_offset_I, self.opt_offset_Q)
        print 'Amplitude imbalance (Q / I) = %0.3f (base amplitude = %0.3f)' % (self.opt_amp_factor, self.base_amplitude)
        print 'Total skew imbalance (rad) = %0.4f' % (self.calculate_sideband_phase(self.opt_skew_ps, total_phase=True))

    def tune_lo(self, verbose=None, plot=True, mode='coarse'):
        print 'tuning LO leakage'

        if not self.use_SH:
            self.spec.set_frequency(self.lo_frequency)
            self.spec.set_rf_on(1)

        if type(mode) == tuple or type(mode) == list:
            vrange,n_it,n_avg = mode
        elif mode == 'coarse':
            vrange = 0.6    # +/- half that. ok
            n_it = 3        #number of sweeps
            n_avg = 1       #number of averages samples per point
        elif mode == 'fine':
            vrange = 0.1
            n_it = 2
            n_avg = 3

        if verbose:
            self.verbose = verbose
        args = (self.awg, self.chans, self.spec)
        if self.use_SH:
            args = tuple(list(args) + [self.lo_frequency])
        m = Minimizer(self.lo_leakage_func, args=args,
                      kwargs={'verbose':self.verbose, 'n_avg':n_avg}, n_it=n_it, n_eval=11,
                      verbose=self.verbose, plot=plot)

        self.get_tuning_parameters() # update current settings

        m.add_parameter(Parameter('vI', value=self.opt_offset_I, vrange=vrange, minstep=0.001))
        m.add_parameter(Parameter('vQ', value=self.opt_offset_Q, vrange=vrange, minstep=0.001))
        m.minimize()
        self.get_tuning_parameters() # update current settings

#        self.reset_valo()
        if not self.use_SH:
            self.spec.set_rf_on(0)

        self.opt_offset_I = m.params['vI'].value
        self.opt_offset_Q = m.params['vQ'].value
        print 'Optimal offsets: %0.03f,%0.03f' % (m.params['vI'].value,m.params['vQ'].value)

        #The minimizer command automatically leaves it in its best values.


    def tune_osb(self, verbose=None, plot=True, mode='coarse',
                 load_previous_vals=False):
        '''
            does not adjust sideband phase to account for awg skew, user must ADD
        '''

        print 'tuning OSB leakage..' # Should be more informative.

        if not self.use_SH:
            self.spec.set_frequency(self.osb_freq)
            self.spec.set_rf_on(1)

        if type(mode) == tuple or type(mode) == list:
            amp_range, skew_range, n_it, n_avg = mode
        elif mode == 'coarse':
            amp_range = 0.5
            skew_range = 3000 #in picoseconds
            n_it = 3        #number of sweeps
            n_avg = 5       #number of averages samples per point
        elif mode == 'fine':
            amp_range = 0.1
            skew_range = 300 #in picoseconds
            n_it = 2        #number of sweeps
            n_avg = 3       #number of averages samples per point

        self.get_tuning_parameters() # update current settings
        args = (self.awg, self.chans, self.spec)
        if self.use_SH:
            args = tuple(list(args) + [self.osb_freq])
        m = Minimizer(self.osb_min_func, args=args,
                      kwargs={'verbose':self.verbose, 'n_avg':n_avg}, n_it=n_it, n_eval=11,
                      verbose=self.verbose, plot=plot)

        m.add_parameter(Parameter('skew', value=self.opt_skew_ps, vrange=skew_range))
        m.add_parameter(Parameter('amplitude', value=self.opt_amp_factor, vrange=amp_range))
        params = m.minimize()
        self.get_tuning_parameters() # update current settings

        if not self.use_SH:
            self.spec.set_rf_on(0)
#        self.reset_valo()

        skew_phase = self.calculate_sideband_phase(self.opt_skew_ps)
        total_skew_rad = self.calculate_sideband_phase(self.opt_skew_ps, total_phase=True)
        print 'Amplitude imbalance: %0.4f' % params['amplitude'].value
        print 'Skew imbalance (ps) and (rad): %0.4f, %0.4f' % \
                (params['skew'].value, skew_phase)
        print 'sideband phase: %0.4f + %0.4f = %0.4f' % \
            (self.sideband_phase, skew_phase, total_skew_rad)

    def calculate_sideband_phase(self, skew_ps, total_phase=False):
        skew_phase = -(2 * np.pi * self.if_freq * skew_ps * 1e-12)
        if total_phase:
            return skew_phase + self.qubit_info.get_sideband_phase()
        else:
            return skew_phase

    def load_test_waveform(self):

        self.awg.delete_all_waveforms()
        self.awg.get_id() #Sometimes AWGs are slow.  If it responds it's ready.

        length = abs(int(1e9/self.if_freq *100))
        qubit_info = mclient.get_qubit_info(self.qubit_info.get_name())

        s = sequencer.Sequence()
        s.append(sequencer.Combined([
                            sequencer.Constant(length, 0.25, chan=qubit_info.sideband_channels[0]),
                            sequencer.Constant(length, 0.25, chan=qubit_info.sideband_channels[1])]))

        s = sequencer.Sequencer(s)

        s.add_ssb(qubit_info.ssb)
        seqs = s.render()
        self.seqs = seqs


        l = awgloader.AWGLoader(bulkload=config.awg_bulkload)
        base = 1
        for i in range(1, 5):
            awg = instruments['AWG%d'%i]
            if awg:
                chanmap = {1:base, 2:base+1, 3:base+2, 4:base+3}
                logging.info('Adding AWG%d, channel map: %s', i, chanmap)
                l.add_awg(awg, chanmap)
                base += 4
        l.load(seqs)
        l.run()