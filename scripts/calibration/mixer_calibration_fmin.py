import matplotlib.pyplot as plt
from scipy.optimize import fmin
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

HISTORY = []

def define_object(obj):
    '''Lets you pass objects either by their name or directly as mclient containers'''
    if type(obj) == str:
        return instruments[obj]
    else:
        return obj

OFFS_SCALE = 1/0.0025
def lo_min_func(params, awg, chans, spec, verbose):
    i_off = max(params[0]*OFFS_SCALE, -0.5)
    i_off = min(i_off, 0.5)
    q_off = max(params[1]*OFFS_SCALE, -0.5)
    q_off = min(q_off, 0.5)

    awg.set({
        'ch%s_offset' % chans[0]: i_off,
        'ch%s_offset' % chans[1]: q_off,
    })
    time.sleep(spec.get_delay())
    val = spec.get_power()
    if verbose:
        print 'Measuring at (%.06f, %.06f): %.01f' % \
                (i_off, q_off, val)
                
    global HISTORY
    HISTORY.append((i_off, q_off, val))
    return val

SKEW_SCALE = 10e5  #Scaling factor for skew tuning (makes NM simpler)
def osb_min_func(params, awg, chans, spec, chan_select, verbose):
    skew = max(params[0]*SKEW_SCALE, -5000)
    skew = min(skew, 5000)
    amp = max(params[1], 0.5)
    amp = min(amp, 1.5)

    if chan_select == 'i':
        ch_tune, ch_base = chans
    elif chan_select == 'q':
        ch_base, ch_tune = chans
    ch_amp = awg.get('ch%s_amplitude' % ch_base)
    awg.set({
        'ch%s_skew'%ch_tune: skew,
        'ch%s_amplitude'%ch_tune: (amp * ch_amp)
    })
    time.sleep(spec.get_delay())
    val = spec.get_power()
    if verbose:
        print 'Measuring at (%.06f, %.06f): %.01f' % \
                (skew, amp, val)
                
    global HISTORY
    HISTORY.append((skew, amp, val))
    return val

class Mixer_Calibration(object):
    '''
    Example code:

    cal = mixer_cal('ancilla', 7759.634e6, 'vspec', 'AWG2', verbose=True,
                        base_amplitude=2.0,
                        va_lo='va_lo_low')

    cal.prep_instruments(reset_offsets=True, reset_ampskew=True)
    cal.tune_lo()
    cal.tune_osb()

    # this function will set the correct qubit_info sideband phase for use in experiments
    #    i.e. combines the AWG skew with the current sideband phase offset
    cal.set_tuning_parameters(set_sideband_phase=True)
    cal.load_test_waveform()
    cal.print_tuning_parameters()
    '''

    def __init__(self, qubit_info, frequency, spec='va_spec',
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

        self.old_valo = self.spec.get_rfsource()
        if self.va_lo is not None:
            self.spec.set_rfsource(self.va_lo)
            print 'setting VALO: %s'% (self.va_lo)

        valo_inst = instruments[self.va_lo]
        valo_inst.set_power(13)
        valo_inst.set_use_extref(True)
        valo_inst.set_pulse_on(False)
        time.sleep(1)

    def reset_valo(self):
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

    def plot(self, title=None):
        global HISTORY
        h = np.array(HISTORY)
        xs, ys, zs = h[:,0], h[:,1], h[:,2]

        zs = np.log(zs - min(zs) + 0.1)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(xs,ys, c=zs, s=200)
        if title:
            ax.set_title(title)
        
    def print_tuning_parameters(self):
        print '-------- SSB Tuning parameters: %s -------' % (self.qubit_info.get_name())
        print 'I, Q offsets = (%0.3f, %0.3f)' % (self.opt_offset_I, self.opt_offset_Q)
        print 'Amplitude imbalance (Q / I) = %0.3f (base amplitude = %0.3f)' % (self.opt_amp_factor, self.base_amplitude)
        print 'Total skew imbalance (rad) = %0.4f' % (self.calculate_sideband_phase(self.opt_skew_ps, total_phase=True))

    def tune_lo(self, x0=[0.0, 0.0], plot=True):
        print 'tuning LO leakage'
        global HISTORY
        HISTORY = []
        
        self.spec.set_frequency(self.lo_frequency)
        self.spec.set_rf_on(1)

        # TODO: Add ability to start from previous values
        xopt = fmin(lo_min_func, x0, xtol=0.01, ftol=0.05, maxfun=50, 
                    args=(self.awg, self.chans,self.spec,self.verbose))

        #Set opt params
        lo_min_func(xopt, self.awg, self.chans, self.spec, 0)
        self.get_tuning_parameters() # update current settings

#        self.reset_valo()
        self.spec.set_rf_on(0)

        print 'Optimal offsets: %0.03f,%0.03f' % (self.opt_offset_I,
                                                  self.opt_offset_Q)
                                                  
        if plot:
            self.plot('LO tuning')

    def tune_osb(self, x0=[0.0, 1.0], chan_select='q', plot=True):
        '''
            does not adjust sideband phase to account for awg skew, user must ADD
        '''
        print 'tuning OSB leakage'
        global HISTORY
        HISTORY = []
        
        self.spec.set_frequency(self.osb_freq)
        self.spec.set_rf_on(1)

        xopt = fmin(osb_min_func, x0, xtol=0.01, ftol=0.05, maxfun=50,
                    args=(self.awg, self.chans, self.spec, chan_select, self.verbose))

        #Set opt params.
        osb_min_func(xopt, self.awg, self.chans, self.spec, chan_select,0)
        self.get_tuning_parameters() # update current settings

        self.spec.set_rf_on(0)
#        self.reset_valo()

        skew_phase = self.calculate_sideband_phase(self.opt_skew_ps)
        total_skew_rad = self.calculate_sideband_phase(self.opt_skew_ps, total_phase=True)
        print 'Amplitude imbalance: %0.4f' % xopt[1]
        print 'Skew imbalance (ps): %0.4f' % self.opt_skew_ps
        print 'sideband phase: %0.4f + %0.4f = %0.4f' % \
            (self.sideband_phase, skew_phase, total_skew_rad)

        if plot:
            self.plot('OSB tuning')
            
    def calculate_sideband_phase(self, skew_ps, total_phase=False):
        skew_phase = -(2 * np.pi * self.if_freq * skew_ps * 1e-12)
        if total_phase:
            return skew_phase + self.qubit_info.get_sideband_phase()
        else:
            return skew_phase

    def load_test_waveform(self):

        self.awg.delete_all_waveforms()
        self.awg.get_id() #Sometimes AWGs are slow.  If it responds it's ready.
            #The above comment isn't necessarily true.

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