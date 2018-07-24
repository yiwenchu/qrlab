# fpgapulses.py, Reinier Heeres, 2014
#
# Some functions to generate pulses suitable for FPGA sequences.

import numpy as np
from pulseseq import sequencer, pulselib, ampgen
sequencer.Pulse.RANGE_ACTION = sequencer.IGNORE
import copy
import load_fpgadevelop
from YngwieEncoding import *
import matplotlib.pyplot as plt

import mclient
ro_ins = mclient.instruments['readout']

REGOPLEN = 48

def RegOp(ins, **kwargs):
    length = kwargs.pop('length', REGOPLEN)
    return sequencer.Constant(length, 0, chan='master', master_register_instruction=ins, **kwargs)

def RegDelay(ireg, **kwargs):
    return sequencer.Constant(REGOPLEN, 0, chan=-1, master_register_delay=ireg)

def RegOpFact(op_str):
    op = getattr(RegisterInstruction, op_str)
    return lambda *args, **kwargs: RegOp(op(*args), **kwargs)

Nop = RegOpFact('NOP')
Mov = RegOpFact('MOV')
MovImm = RegOpFact('MOVI')
Add = RegOpFact('ADD')
AddImm = RegOpFact('ADDI')
Mult = RegOpFact('MULT')
MultImm = RegOpFact('MULTI')

def start_loop(name, n):
    return Nop(label=name, master_counter0=n)

def end_loop(name):
    return Nop(label=name, master_counter0=-1, master_internal_function=FPGASignals.c0, jump=('next', name))

def FPGARange(seq, reg_ranges, label, inner=False):
    begin_seq = []
    counter = 'master_counter%d' % (1 if inner else 0)
    # Decrement counter at end of loop
    end_seq = [Nop(**{counter:-1})]
    # Jump to seq[0] for each iteration
    seq[0].set_params(label=label)
    N = len(reg_ranges)
    for i, (reg_n, start, stop, steps) in enumerate(reg_ranges):
        delta = (stop - start) / steps
        begin_kwargs, end_kwargs = {}, {}
        # Set the Register to its starting value
        # Initialize the loop counter once
        if i == 0:
            begin_kwargs[counter] = steps
        begin_seq.append(MovImm(reg_n, start, **begin_kwargs))

        # Add step to register at end of loop
        # conditionally jump based on counter status after last update
        if i == N:
            end_kwargs['master_internal_function'] = \
                FPGASignals.c1 if inner else FPGASignals.c0
            end_kwargs['jump'] = ('next', label)
        end_seq.append(AddImm(reg_n, delta))

    return begin_seq + seq + end_seq


def MeasurementPulse(pulselen=200, delay=0, ro_delay=0, **kwargs):
    label = kwargs.pop('label', None)
    cb = sequencer.Combined([
        sequencer.Constant(pulselen, 2, chan='m0'),
        sequencer.Sequence([
            sequencer.Delay(delay),
            sequencer.Constant(40, 1, chan='master', **kwargs),
            sequencer.Constant(220, 2, chan='master', **kwargs)
        ]),
    ], label=label, align=sequencer.ALIGN_LEFT)
    if ro_delay == 0:
        return cb
    else:
        return sequencer.Sequence([sequencer.Delay(ro_delay), cb])

def long_delay(length, block_size, label):
    s = sequencer.Sequence()
    if (length % block_size) != 0:
        raise Exception('Length not an integer multiple of block size')
    nblocks = length / block_size
    if nblocks > 2**17:
        raise Exception('Too many blocks requested!')
    s.append(RegOp(RegisterInstruction.MOVI(4, nblocks), label='init_%s'%label))
    s.append(RegOp(RegisterInstruction.ADDI(4, -1), length=block_size, label='loop_%s'%label))
    s.append(RegOp(RegisterInstruction.CMPI(4, 0),
                              master_internal_function=FPGASignals.r0,
                              jump=('done_%s'%label, 'loop_%s'%label)))
    s.append(sequencer.Delay(256, label='done_%s'%label))
    return s

def LongMeasurementPulse(pulselen=None, rolen=None, reflen=None, ro_delay=0, pulse_pump=False, **kwargs):
    sequencer.Pulse.RANGE_ACTION = sequencer.IGNORE
    label = kwargs.pop('label', None)
    if pulselen is None:
        pulselen = ro_ins.get_pulse_len()
    if rolen is None:
        rolen = ro_ins.get_acq_len()
    if reflen is None:
        reflen = ro_ins.get_ref_len()

    ro = sequencer.Constant(reflen, 3, chan='master', **kwargs)
    kwargs.pop('master_integrate', None)   # Remove integrate from kwargs
    if reflen != rolen:
        ro = sequencer.Sequence([ro, sequencer.Constant(rolen-reflen, 2, chan='master', **kwargs)])

    if pulse_pump:
        cb = sequencer.Sequence([
            sequencer.Constant(800, 1, chan='m1'),
            sequencer.Combined([
                sequencer.Constant(pulselen, 2, chan='m0'),
                sequencer.Constant(rolen, 1, chan='m1'),
                ro,
                ], align=sequencer.ALIGN_LEFT)
            ], label=label)
    else:
        cb = sequencer.Combined([
            sequencer.Constant(pulselen, 2, chan='m0'),
            ro,
        ], label=label, align=sequencer.ALIGN_LEFT)

    if ro_delay == 0:
        return cb
    else:
        return sequencer.Sequence([sequencer.Delay(ro_delay), cb])

class FPGAAmplitudeRotation(object):
    '''
    A rotation with an angle that is controlled using the pulse amplitude.
    Both Pi and pi/2 amplitude can be specified and the amplitude for any
    given amplitude will be interpolated.
    '''

    def __init__(self, Ipulse, Qpulse, pi_amp, pi2_amp=0, chans=(0,1), marker_chan='m1', marker_val=2, marker_pad=0):
        if marker_pad != 0:
            mp = np.zeros(marker_pad)
            Ipulse = sequencer.Pulse(Ipulse.name+'_pad%s'%len(mp), data=np.concatenate([mp,Ipulse.data,mp]))
            sequencer.Sequence([sequencer.Delay(marker_pad), Ipulse])
        if Ipulse:
            Ipulse = Ipulse.rendered()
        if Qpulse:
            Qpulse = Qpulse.rendered()
        self.Ipulse = Ipulse
        self.Qpulse = Qpulse
        self.ampgen = ampgen.AmpGen()
        self.set_pi_amp(pi_amp, pi2_amp)
        self.chans = chans
        self.marker_chan = marker_chan
        self.marker_val = marker_val
        self.marker_pad = marker_pad
        super(FPGAAmplitudeRotation, self).__init__()

    def set_pi_amp(self, pi_amp, pi2_amp=0):
        self.pi_amp = pi_amp
        self.pi2_amp = pi2_amp
        if pi2_amp != 0:
            self.ampgen.set_amp_spec([pi2_amp, pi_amp])
        else:
            self.ampgen.set_amp_spec(pi_amp)

    def __call__(self, alpha, phase, amp=None, **kwargs):
        '''
        Generate a rotation pulse of angle <alpha> around axis <phase>.
        If <amp> is specified that amplitude is used and <alpha> is ignored.
        '''
        if amp is None:
            amp = self.ampgen(alpha)
        amp *= 65535.0
        pulses = []

        kwargs['chan%d_mixer_amplitudes'%self.chans[0]] = (np.cos(phase)*amp, np.sin(phase)*amp)
        kwargs['chan%d_mixer_amplitudes'%self.chans[1]] = (-np.sin(phase)*amp, np.cos(phase)*amp)
        if 'detune_period' in kwargs and kwargs['detune_period'] != 0:
            period = kwargs.pop('detune_period')
            ts = np.arange(len(self.Ipulse.data))
            data = self.Ipulse.data * np.exp(2j * np.pi * ts / period)
            if 0:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(ts, np.real(data))
                plt.plot(ts, np.imag(data))
                plt.suptitle('period=%s'%period)
            Ipulse = sequencer.Pulse('%s_re%s'%(self.Ipulse.name, period), np.real(data), chan=self.chans[0]).rendered()
            Qpulse = sequencer.Pulse('%s_im%s'%(self.Ipulse.name, period), np.imag(data), chan=self.chans[1]).rendered()
            Ipulse.set_params(**kwargs)
            pulses.append(Ipulse)
            Qpulse.set_params(**kwargs)
            pulses.append(Qpulse)

        else:
            if self.Ipulse:
                p = copy.deepcopy(self.Ipulse)
                pulses.append(p)
                pulses[-1].chan = self.chans[0]
                pulses[-1].set_params(**kwargs)
                if not self.Qpulse:
                    pulses.append(sequencer.Constant(self.Ipulse.get_length(), 1e-6, chan=self.chans[1]))
                    pulses[-1].set_params(mixer_amplitudes=(-amp*np.sin(phase), amp*np.cos(phase)), **kwargs)
            if self.Qpulse:
                p = copy.deepcopy(self.Qpulse)
                pulses.append(p)
                pulses[-1].chan = self.chans[1]
                pulses[-1].set_params(**kwargs)

        sequencer.Pulse.RANGE_ACTION = sequencer.IGNORE
        if self.marker_chan is not None:
            mpulse = sequencer.Constant(pulses[0].get_length(), self.marker_val, chan=self.marker_chan, **kwargs)
            pulses.append(mpulse)

        if len(pulses) == 1:
            return sequencer.Sequence(pulses)
        else:
            return sequencer.Combined(pulses, **kwargs)

class FPGADetunedSum(object):
    '''
    A rotation with an angle that is controlled using the pulse amplitude.
    Both Pi and pi/2 amplitude can be specified and the amplitude for any
    given amplitude will be interpolated.
    '''

    def __init__(self, pulse, chans=(0,1), marker_chan='m1', marker_val=2, f0s=None, amps=None, starkcorr=True):
        pulse = pulse.rendered()
        self.pulse = pulse
        self.chans = chans
        self.marker_chan = marker_chan
        self.marker_val = marker_val
        self.info = []
        self.starkcorr = starkcorr
        self.corr = {}
        self.zero_info = []
        if f0s is not None:
            self.info = zip(f0s, amps)
        super(FPGADetunedSum, self).__init__()

    def add(self, f, amp):
        self.info.append((f, amp))

    def add_zero(self, f):
        self.zero_info.append(f)

    def f_to_period(self, f):
        if f == 0:
            return 1e50
        else:
            return 1e9 / f

    def determine_correction(self):
        base_data = self.pulse.data
        data = self.get_pulse_data()
        ts = np.arange(len(data)) - len(data) / 2
        self.corr = {}
        for ipeak, (f0, amp) in enumerate(self.info):
            period = self.f_to_period(f0)
            refdata = amp * base_data * np.exp(1j * (2 * np.pi * ts / period))
            refcoeff = np.sum(refdata * np.exp(-1j * (2 * np.pi * ts / period))) / len(data)
            coeff = np.sum(data * np.exp(-1j * (2 * np.pi * ts / period))) / len(data)
            self.corr[ipeak] = np.real(refcoeff)/np.real(coeff)

    def get_stark_pulse_data(self):
        pulse = self.pulse.data
        f0s = [i[0] for i in self.info]
        amps = [i[1] for i in self.info]

        avgpulse = np.sum(pulse) / len(pulse)
        drive_scales = 0.5 / (avgpulse * len(pulse) * 1e-9)

        d_t = np.zeros([len(pulse)+1, len(amps)])
        for i_t in range(len(pulse)):
            damp = pulse[i_t] * drive_scales

            # Resulting detuning on each peak at this point in time
            for i_tgt in range(len(amps)):
                for i_src in range(len(amps)):
                    if i_tgt == i_src:
                        continue
                    d_t[i_t+1,i_tgt] += damp**2 / (f0s[i_tgt] + d_t[i_t,i_tgt] - f0s[i_src] - d_t[i_t,i_src])

        f_t = f0s + d_t[1:,:]
        if 0:
            plt.figure()
            for i in range(len(amps)):
                plt.plot(f_t[:,i]/1e6, label='F%d'%i)
            plt.legend()

        phi_t = np.zeros_like(f_t)
        for i in range(len(amps)):
            phi_t[:,i] = np.cumsum(f_t[:,i] * 2 * np.pi * 1e-9)
        data = np.zeros_like(pulse, dtype=np.complex)
        for i in range(len(amps)):
            phi0 = phi_t[int(round(len(pulse)/2)),i]
            data += pulse * amps[i] *  np.exp(1j * (phi_t[:,i] - phi0))
        return data

    def get_pulse_data(self):
        if self.starkcorr:
            return self.get_stark_pulse_data()

        # Generate base pulse
        base_data = self.pulse.data
        data = np.zeros_like(base_data, dtype=np.complex)
        for ipeak, (f0, amp) in enumerate(self.info):
            period = self.f_to_period(f0)
            corr = self.corr.get(ipeak, 1)
            ts = np.arange(len(base_data)) - len(base_data) / 2
            data += corr * amp * base_data * np.exp(1j * (2 * np.pi * ts / period))

        # Add cancellation drives
        for zero_period in self.zero_info:
            corr = np.exp(-2j * np.pi * ts / zero_period)
            coeff = np.sum(data * corr)
            data -= coeff/len(data) * np.exp(2j * np.pi * ts / zero_period)

        return data

    def get_pulse_name(self):
        return ''.join(['p%sa%s'%(period, amp) for period, amp in self.info])

    def __call__(self, alpha, amp=None, plot=False, **kwargs):
        '''
        Generate a rotation pulse of angle <alpha> around axis <phase>.
        If <amp> is specified that amplitude is used and <alpha> is ignored.
        '''

        self.determine_correction()

        sequencer.Pulse.RANGE_ACTION = sequencer.IGNORE
        data = self.get_pulse_data()
        name = self.get_pulse_name()
        Ipulse = sequencer.Pulse('%s_I'%name, np.real(data), chan=self.chans[0]).rendered()
        Qpulse = sequencer.Pulse('%s_Q'%name, np.imag(data), chan=self.chans[1]).rendered()
        if plot:
            plt.figure()
            plt.plot(self.pulse.data)
            plt.plot(np.real(data))
            plt.plot(np.imag(data))
            plt.plot(2*np.arange(len(data)/2), np.real(data[::2]), 'ks')
            plt.plot(2*np.arange(len(data)/2), np.imag(data[::2]), 'ro')

        pulses = []
        Ipulse.set_params(mixer_amplitudes=(0xffff, 0), **kwargs)
        pulses.append(Ipulse)
        Qpulse.set_params(mixer_amplitudes=(0, 0xffff), **kwargs)
        pulses.append(Qpulse)

        if self.marker_chan is not None:
            mpulse = sequencer.Constant(pulses[0].get_length(), self.marker_val, chan=self.marker_chan, **kwargs)
            pulses.append(mpulse)

        if len(pulses) == 1:
            return sequencer.Sequence(pulses, **kwargs)
        else:
            return sequencer.Combined(pulses, **kwargs)

class FPGAMarkerLenRotation(object):

    def __init__(self, pi_len, chan='m0', val=1):
        self.pi_len = pi_len
        self.chan = chan
        self.val = val
        super(FPGAMarkerLenRotation, self).__init__()

    def __call__(self, alpha, phase, amp=None, **kwargs):
        '''
        Generate a rotation pulse of angle <alpha> around axis <phase>.
        If <amp> is specified that amplitude is used and <alpha> is ignored.
        '''

        plen = int(round(self.pi_len * alpha / np.pi / 4.0) * 4)
        print 'Alpha=%s --> plen = %s' % (alpha, plen)
        if plen < 24:
            print 'Warning: pulse length < 24!'
            plen = 24
        return sequencer.Constant(plen, self.val, chan=self.chan)