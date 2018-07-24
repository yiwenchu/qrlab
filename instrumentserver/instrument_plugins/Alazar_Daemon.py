# Alazar acquisition daemon.
#
# This daemon provides 2 shared objects:
# - AlazarCard, to directly set channel parameters
# - AlazarDaemon, to manage buffers and perform the actual acquisition

import sys

import ctypes
import types
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['legend.fontsize'] = 8
import time
from lib.dll_support import alazar
AC = alazar.Constants
from lib.math import demod
from instrument import Instrument
import logging
import gc
import os

import objectsharer as objsh
#objsh.logger.setLevel(logging.DEBUG)

class Alazar_Daemon(Instrument):

    def __init__(self, name, systemid=1, boardid=1, **kwargs):
        super(Alazar_Daemon, self).__init__(name)

        self._systemid = systemid
        self._boardid = boardid
        self._allocated_id = None
        self._bufs = []
        self._start_bufs = []
        self._interrupt = False
        self._capturing = False
        self._card = alazar.Alazar(systemid, boardid)

        self.add_parameter('if_period', type=types.IntType,
                           flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                           minval=2, maxval=1000, value=20,
                           help='Intermediate Frequency period')
        self.add_parameter('weight_func', type=types.StringType,
                           flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                           help='''
Weight function file, either a .npy (numpy file) or a txt file.
The data should have length #IF periods. If real valued the same weight
function is applied to both I and Q quadratures, if complex valued the
real part is applied to I and the imaginary part to Q.
''')
        self.add_parameter('nbuffers', type=types.IntType,
                           flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                           minval=1, maxval=512, value=4,
                           set_func=lambda x: True,
                           help='Number of acquisition buffers')
        self.add_parameter('nrecperbuf', type=types.IntType,
                           flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                           minval=1, maxval=1e4, value=1,
                           set_func=lambda x: True,
                           help='Number of records per buffer')
        self.add_parameter('nsamples', type=types.IntType,
                           flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                           minval=64, maxval=1e9, value=5120,
                           help='Number of samples per record')
        self.add_parameter('naverages', type=types.IntType,
                           flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                           minval=1, maxval=1000000, value=500,
                           set_func=lambda x: True,
                           help='Number of averages to do')
        self.add_parameter('ntotal_rec', type=types.IntType,
                           flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                           value=0,
                           set_func=lambda x: True,
                           help='Total number of records that is to be acquired')
        self.add_parameter('timeout', type=types.FloatType,
                           flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                           minval=100, maxval=600000, value=30000, units='msec',
                           set_func=lambda x: True,
                           help='Timeout for acquisition, in msec')

        self.add_parameter('channels', type=types.IntType,
                           flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                           format_map={
                               AC.CHANNEL_A: 'Chan A',
                               AC.CHANNEL_B: 'Chan B',
                               AC.CHANNEL_AB: 'Chan AB',
                           }, value=AC.CHANNEL_AB,
                           set_func=lambda x: True)
        self.add_parameter('coupling', flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                type=types.IntType, channels=(1,2), channel_prefix='ch%d_',
                format_map={
                    AC.COUPLING_AC: 'AC',
                    AC.COUPLING_DC: 'DC'
                }, value=AC.COUPLING_DC,
#                gui_group='channels',
                set_func=lambda x, **y: True)
        self.add_parameter('impedance', flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                type=types.IntType, channels=(1,2), channel_prefix='ch%d_',
                format_map={
                    AC.IMPEDANCE_1k: '1k',
                    AC.IMPEDANCE_50: '50',
                    AC.IMPEDANCE_75: '75',
                    AC.IMPEDANCE_300: '300',
                }, value=AC.IMPEDANCE_50,
                units='Ohm',
                gui_group='channels',
                set_func=lambda x, **y: True)
        self.add_parameter('bandwidth_limit', flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                type=types.BooleanType, channels=(1,2), channel_prefix='ch%d_',
                value=False,
                gui_group='channels')
        self.add_parameter('range', flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                type=types.IntType, channels=(1,2), channel_prefix='ch%d_',
                format_map={
                    AC.RANGE_0040: '40mV',
                    AC.RANGE_0100: '100mV',
                    AC.RANGE_0200: '200mV',
                    AC.RANGE_0400: '400mV',
                    AC.RANGE_1: '1V',
                    AC.RANGE_2: '2V',
                    AC.RANGE_4: '4V',
                }, value=AC.RANGE_1,
                units='V',
 #               gui_group='channels',
                set_func=lambda x, **y: True)

        # clock and sampling parameters
        self.add_parameter('clock_source', flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                type=types.IntType, format_map={
                    AC.CLKSRC_INT: 'INT',
                    AC.CLKSRC_FASTEXT: 'FASTEXT',
                    AC.CLKSRC_MEDEXT: 'MEDEXT',
                    AC.CLKSRC_SLOWEXT: 'SLOWEXT',
                    AC.CLKSRC_EXT_10M: 'EXT10M',
                }, value=AC.CLKSRC_INT)
        self.add_parameter('sample_rate', flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                units='Hz', type=types.IntType, format_map={
                    AC.SR_1k: '1k',
                    AC.SR_2k: '2k',
                    AC.SR_5k: '5k',
                    AC.SR_10k: '10k',
                    AC.SR_20k: '20k',
                    AC.SR_50k: '50k',
                    AC.SR_100k: '100k',
                    AC.SR_1M: '1M',
                    AC.SR_2M: '2M',
                    AC.SR_5M: '5M',
                    AC.SR_10M: '10M',
                    AC.SR_20M: '20M',
                    AC.SR_50M: '50M',
                    AC.SR_100M: '100M',
                    AC.SR_250M: '250M',
                    AC.SR_500M: '500M',
                    AC.SR_1G: '1G',
                    AC.SR_1G_FROM_EXT10: '1GEXT10',
                }, value=AC.SR_1G)
        self.add_parameter('clock_edge', flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                type=types.IntType, format_map={
                    AC.CLKEDG_RISE: 'Rise',
                    AC.CLKEDG_FALL: 'Fall',
                }, value=AC.CLKEDG_RISE)

        # add trigger parameters
        self.add_parameter('trig_op', flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                type=types.IntType, format_map={
                    AC.TRIG_ENGINE_OP_J: 'J',
                    AC.TRIG_ENGINE_OP_K: 'K',
                    AC.TRIG_ENGINE_OP_J_OR_K: 'J|K',
                    AC.TRIG_ENGINE_OP_J_AND_K: 'J&K',
                    AC.TRIG_ENGINE_OP_J_XOR_K: 'J^K',
                    AC.TRIG_ENGINE_OP_J_AND_NOT_K: 'J&!K',
                    AC.TRIG_ENGINE_OP_NOT_J_AND_K: '!J&K',
                }, value=AC.TRIG_ENGINE_OP_J,
                gui_group='trigger')
        self.add_parameter('trig_slope', flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                type=types.IntType, format_map={
                    AC.TRIG_SLP_POS: 'POS',
                    AC.TRIG_SLP_NEG: 'NEG',
                }, value=AC.TRIG_SLP_POS,
                channels=('J','K'), channel_prefix='eng%s_',
                gui_group='trigger')
        self.add_parameter('trig_src', flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                type=types.IntType, format_map={
                    AC.TRIG_CHAN_A: 'A',
                    AC.TRIG_CHAN_B: 'B',
                    AC.TRIG_EXTERNAL: 'EXT',
                    AC.TRIG_DISABLE: 'DISABLE',
                    AC.TRIG_CHAN_C: 'C',
                    AC.TRIG_CHAN_D: 'D',
                }, value=AC.TRIG_DISABLE,
                channels=('J','K'), channel_prefix = 'eng%s_',
                gui_group='trigger')
        self.add_parameter('trig_lvl', flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                type=types.IntType, channels=('J','K'), channel_prefix='eng%s_',
                minval=0, maxval=255,
                gui_group='trigger')
        self.add_parameter('ext_trig_delay', flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                units='samples', type=types.IntType, value=0,
                gui_group='trigger')
        self.add_parameter('ext_trig_timeout', flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                units='sec', type=types.FloatType,
                gui_group='trigger')
        self.add_parameter('ext_trig_coupling', flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                type=types.IntType, format_map={
                    AC.COUPLING_AC: 'AC',
                    AC.COUPLING_DC: 'DC'
                }, value=AC.COUPLING_DC,
                gui_group='trigger')
        self.add_parameter('ext_trig_range', flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                type=types.IntType, unit='V', format_map={
                    AC.ETR_5V: 5,
                    AC.ETR_1V: 1,
                }, value=AC.ETR_5V,
                gui_group='trigger')

        self.add_parameter('real_signals', flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                type=types.BooleanType, value=True,
                help='Whether to convert complex voltages to real signals')
        self.add_parameter('signal_phase', flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
                type=types.FloatType, value=-1,
                help='The signal phase to use for converting from complex to real signals')

        self.add_function("setup_channels")
        self.add_function("setup_clock")
        self.add_function("setup_trigger")

        if kwargs.pop('reset', False):
            self.reset()
        else:
            self.get_all()

        self.set(kwargs)
        self.allocate_buffers()

    def set_interrupt(self, val):
        if val:
            logging.info('Setting capture interrupt flag')
        self._interrupt = val

    def get_interrupt(self):
        return self._interrupt

    # We use the FLAG_SOFTGET feature to read these values

    def do_set_nsamples(self, val):
        if (val % 64):
            raise ValueError('Number of samples should be multiple of 64')

    def do_set_bandwidth_limit(self, val, channel):
        self._card.set_ch_bw_limit(channel, val)

    def _get_nchannels(self):
        if self.get_channels() == AC.CHANNEL_AB:
            return 2
        else:
            return 1

    def setup_clock(self):
        logging.debug('Setting up clock')
        logging.debug('src: %s, sr: %s, edge: %s', self.get_clock_source(), self.get_sample_rate(), self.get_clock_edge())
        err = alazar.ats.AlazarSetCaptureClock(self._card.handle,
                self.get_clock_source(),
                self.get_sample_rate(),
                self.get_clock_edge(),
                1)
        alazar.CHK(err)

    def setup_channels(self):
        logging.debug('Setting up channels')
        logging.debug('ch1_range: %s, coupling: %s, impedance: %s', self.get_ch1_range(), self.get_ch1_coupling(), self.get_ch1_impedance())
        self._card.set_capture_channels(self.get_channels())
        self._card.set_ch_props(
            AC.CHANNEL_A, self.get_ch1_range(),
            self.get_ch1_coupling(), self.get_ch1_impedance()
        )
        self._card.set_ch_props(
            AC.CHANNEL_B, self.get_ch2_range(),
            self.get_ch2_coupling(), self.get_ch2_impedance()
        )

    def setup_trigger(self):
        logging.debug('Setting up trigger')
        logging.debug('Op: %s, engj src: %s, slope %s, lvl %s', self.get_trig_op(), self.get_engJ_trig_src(), self.get_engJ_trig_slope(), self.get_engJ_trig_lvl())
        err = alazar.ats.AlazarSetTriggerOperation(self._card.handle,
            self.get_trig_op(),
            AC.TRIG_ENGINE_J, self.get_engJ_trig_src(), self.get_engJ_trig_slope(), self.get_engJ_trig_lvl(),
            AC.TRIG_ENGINE_K, self.get_engK_trig_src(), self.get_engK_trig_slope(), self.get_engK_trig_lvl(),
        )
        alazar.CHK(err)
        logging.debug('Ext coupl %s, ext range %s, delay %s', self.get_ext_trig_coupling(), self.get_ext_trig_range(), self.get_ext_trig_delay())
        err = alazar.ats.AlazarSetExternalTrigger(self._card.handle,
            self.get_ext_trig_coupling(), self.get_ext_trig_range())
        alazar.CHK(err)
        err = alazar.ats.AlazarSetTriggerDelay(self._card.handle, self.get_ext_trig_delay())
        alazar.CHK(err)

    def load_weight_func(self, fn):
        if fn is None or fn == '':
            return 1

        ext = os.path.splitext(fn)[1]
        if ext == '.npy':
            data = np.load(fn)
        elif ext in ('.txt', '.gz', '.bz2'):
            data = np.loads(fn)
        else:
            logging.warning('Unable to load file %s' % fn)
            return 1

        if len(data) != self.get_nsamples() / self.get_if_period():
            raise ValueError('Weight function not of right length')
        return data

    def set_demod(self, avg_periods=1, weight_func=1, bufsize=None):
        '''
        Sets up demodulators.
        <avg_periods> only applies to channel B (the reference), as we might
        want to use weight functions to the corrected shots.
        '''

        if bufsize is None:
            bufsize = self.get_nsamples() * self.get_nrecperbuf()

        # Default is no weighting function.
        self._Iweight = None
        self._Qweight = None

        weight_func = self.load_weight_func(self.get_weight_func())
        if type(weight_func) is np.ndarray:
            if weight_func.dtype in (np.complex, np.complex64, np.complex128):
                self._Iweight = np.real(weight_func)
                self._Qweight = np.imag(weight_func)
            else:
                self._Iweight = weight_func
                self._Qweight = weight_func


        self._demodA = demod.DemodulatorComplex(bufsize, self.get_if_period(), avg_periods=1)
        self._demodB = demod.DemodulatorComplex(bufsize, self.get_if_period(), avg_periods=avg_periods)

        # Garbage collect old demodulators
        gc.collect()

    def allocate_buffers(self):
        '''
        Allocate processing buffers.
        These are multiprocessing safe.
        '''

        recperbuf = self.get_nrecperbuf()
        nbuf = self.get_nbuffers()
        samples = self.get_nsamples()
        nchan = self._get_nchannels()

        if (samples % 64) != 0:
            raise ValueError('Buffer size should be multiple of 64!')

        # Don't reallocate if we already have this size.
        bufsid = (recperbuf, nbuf, samples, nchan)
        if self._allocated_id == bufsid:
            return
        self._allocated_id = bufsid

        bufsize = recperbuf * samples * nchan

        # Make 128-byte aligned arrays
        self._bufs = []
        self._start_bufs = [np.zeros([bufsize+256,], dtype=np.dtype(np.uint8)) for i in range(nbuf)]
        for b in self._start_bufs:
            idx = -b.ctypes.data % 128
            idx += 128
            self._bufs.append(b[idx:idx+bufsize])

        # Garbage collect old buffer
        gc.collect()

# Not guaranteed to be aligned properly
#import multiprocessing as mp
#        mpars = [mp.Array(ctypes.c_uint8, recperbuf*samples*nchan+64) for i in range(nbuf)]
#        self._bufs = [np.frombuffer(buf.get_obj(), dtype=np.dtype(np.uint8)) for buf in mpars]

    def prepare_capture(self):
        print 'Prepare capture: samples: %s, recperbuf: %s, total rec:%s' % (self.get_nsamples(), self.get_nrecperbuf(), self.get_ntotal_rec(),)
        self._card.prepare_capture(self.get_nsamples(), self.get_nrecperbuf(), self.get_ntotal_rec(), self.get_ext_trig_delay(), 0)

    def arm(self):
        self._card.start_capture()

    def break_records(self, N, blocksize=1000, nsamples=None):
        '''
        Break number of records <N> into n * <N'> to use multiple buffers.
        If <nsamples> is specified it is used to estimate a reasonable
        maximum block size.
        '''

        if nsamples is not None:
            blocksize = min(1000, np.ceil(10e6 / nsamples))

        nbufs = 1
        while N > blocksize:
            if (N % 5) == 0:
                nbufs *= 5
                N /= 5
            elif (N % 2) == 0:
                nbufs *= 2
                N /= 2
            else:
                raise ValueError('Please make sure ntotalrec can be factored into <x>*2^y*5^z, with x < 200' )
        return nbufs, N

    def start_capture(self):
        if self._capturing:
            raise Exception('Already capturing, use set_interrupt to stop!')
        self._card.start_capture()
        self._capturing = True
        self.emit('start-capture')

    def end_capture(self):
        self._card.end_capture()
        if self._capturing:
            self.emit('end-capture')
        self.set_interrupt(False)
        self._capturing = False

    def setup_shots(self, N):
        '''
        Setup measurement for <N> single shots.
        '''
        self.end_capture()
        nbufs, N = self.break_records(N)
        self.set_nrecperbuf(N)
        if nbufs != 1:
            self.set_nbuffers(min(4, nbufs))
        else:
            self.set_nbuffers(1)
        self.set_ntotal_rec(nbufs * N)
        self.allocate_buffers()
        self.set_demod(avg_periods=1)
        self.prepare_capture()
        self._card.post_buffers(self._bufs)
        self.start_capture()

    def complex_signal_to_real(self, buf):
        '''
        Convert a complex signal to real. This is done by taking the average
        value of <buf> and computing it's angle in the complex plane. The
        values are then rotated along the real axis and the real part is
        returned.
        '''
        avg = np.average(buf)
        corr = np.exp(-1j * np.angle(avg))
        return np.real(corr * buf)

    def convert_signal(self, buf):
        if self.get_real_signals():
            return self.complex_signal_to_real(buf)
        else:
            return buf

    def take_raw_shots(self, buftimeout=10000):
        '''
        Acquire <N> raw shots.
        Not setup to do more than a 1000.
        '''
        buf = self.get_next_buffer(buftimeout)
        self.end_capture()
        return buf

    def setup_demod_shots(self, N):
        self.setup_avg_shot(N)

    def take_demod_shots(self, acqtimeout=None):
        '''
        Acquire <N> demodulated shots.
        Takes weighting function into account.
        '''

        if acqtimeout is None:
            acqtimeout = self.get_timeout()
        N = self.get_ntotal_rec()
        Nperbuf = self.get_nrecperbuf()
        nsamples = self.get_nsamples()
        periods = nsamples / self.get_if_period()
        i = 0

        IQr = np.zeros([N, periods], dtype=np.complex)
        while i < N:
            buf = self.get_next_buffer(acqtimeout)

            self._demodA.demodulate(buf[:Nperbuf*nsamples])
            IQA = self._demodA.IQ.reshape([Nperbuf, periods])

            # Calculate reference angles
            self._demodB.demodulate(buf[Nperbuf*nsamples:])
            IQB = self._demodB.IQ.reshape([Nperbuf, periods])
            refs = np.exp(-1j * np.angle(np.average(IQB, 1)))
            IQA = IQA * refs[:, np.newaxis]

            # Use weighting function
            if self._Iweight is not None:
                IQr[i:i+Nperbuf,:] = np.real(IQA) * self._Iweight[np.newaxis,:] + 1j * (np.imag(IQA) * self._Qweight[np.newaxis,:])
            else:
                IQr[i:i+Nperbuf,:] = IQA

            self._card.post_buffers(buf)
            i += Nperbuf

        self.end_capture()

        return self.convert_signal(np.array(IQr))

    def setup_avg_shot(self, N):
        '''
        Setup measurement for averaging <N> demodulated shots.
        '''
        self.end_capture()
        nbufs, N = self.break_records(N, nsamples=self.get_nsamples())
        self.set_nrecperbuf(N)
        if nbufs != 1:
            self.set_nbuffers(min(4, nbufs))
        else:
            self.set_nbuffers(1)
        self.set_ntotal_rec(nbufs * N)
        self.allocate_buffers()
        self.set_demod(avg_periods=1)

        self.prepare_capture()
        self._card.post_buffers(self._bufs)
        self.start_capture()

    def get_next_buffer(self, timeout):
        '''
        Get next buffer from Alazar.
        Split timeout in parts < 1 sec and run objectsharer main loop in
        between to check whether an interrupt is requested
        '''

        N = 1
        while timeout > 1000:
            timeout /= 2
            N *= 2

        for i in range(N):
            if objsh.helper.backend:
                objsh.helper.backend.main_loop(0)

            if self.get_interrupt():
                self.end_capture()
                logging.info('Capture interrupted')
                raise Exception('Capture interrupted')

            buf = self._card.get_next_buffer(timeout=timeout)
            if buf is not None:
                return buf

        if buf is None:
            self.end_capture()
            logging.info('Capture timed out')
            raise Exception('Capture timed out')

        return buf

    def take_avg_shot(self, acqtimeout=None):
        '''
        returns one IQ pair per IF period averaged over total number of records.
        Doesn't apply weighting function.
        '''

        if acqtimeout is None:
            acqtimeout = self.get_timeout()
        avg = None
        N = self.get_ntotal_rec()
        Nperbuf = self.get_nrecperbuf()
        nsamples = self.get_nsamples()
        periods = nsamples / self.get_if_period()
        i = 0

        while i < N:
            buf = self.get_next_buffer(acqtimeout)

            self._demodA.demodulate(buf[:Nperbuf*nsamples])
            IQA = self._demodA.IQ.reshape([Nperbuf, periods])

            # Calculate reference angles
            self._demodB.demodulate(buf[Nperbuf*nsamples:])
            IQB = self._demodB.IQ.reshape([Nperbuf, periods])
            refs = np.exp(-1j * np.angle(np.average(IQB, 1)))

            if avg is None:
                avg = np.zeros_like(IQA[0,:])
            for j in range(Nperbuf):
                avg += IQA[j,:] * refs[j]

            self._card.post_buffers(buf)
            i += Nperbuf

        self.end_capture()
        if avg is None:
            return None

        avg /= N
        return self.convert_signal(avg)

    def setup_experiment(self, cycles, weight_func=1):
        self.end_capture()

        if cycles < 100:
            self.set_nbuffers(8)
        else:
            self.set_nbuffers(4)

        self._cycles = cycles
        self._navg = self.get_naverages()

        # Determine how often to repeat #cycles per buffer to get a decent
        # number of shots per buffer
        min_recperbuf = 50
        cyclereps = 1
        while cycles * cyclereps < min_recperbuf:
            if ((self._navg / cyclereps) % 2) == 0:
                cyclereps *= 2
            elif ((self._navg / cyclereps) % 5) == 0:
                cyclereps *= 5
            else:
                raise ValueError('Unable to make cyclelength > %d, please make number of averages divisible by 2 and 5' % min_recperbuf)

        totrec = cycles * self._navg
        recperbuf = cycles * cyclereps
        self._cyclereps = cyclereps
        logging.info('Setup experiment: cycle repetitions %d, recs per buf %d, total records %d',
                     cyclereps, recperbuf, totrec)

        self.set_nrecperbuf(recperbuf)
        self.set_ntotal_rec(totrec)
        self.allocate_buffers()

        periods = self.get_nsamples() / self.get_if_period()
        self.set_demod(avg_periods=periods, weight_func=weight_func)
        self.prepare_capture()
        self._card.post_buffers(self._bufs)
        self.start_capture()

    def update_averages(self, avg_buf, IQ_sum, n):
        try:
            avg_buf[:] = self.convert_signal(IQ_sum / float(n))
            avg_buf.set_attrs(averages=n)
        except Exception, e:
            self._card.end_capture()
            msg = 'Unable to store averages: %s' % str(e)
            logging.warning(msg)
            raise Exception(msg)

    def get_IQ_rel(self, buf, cycles):
        '''
        Return relative IQ values for every shot (1 IQ / shot).
        This takes into account weighting functions
        '''

        blen = len(buf)
        self._demodA.demodulate(buf[:blen/2])
        self._demodB.demodulate(buf[blen/2:])
        phase_corr = np.exp(-1j * np.angle(self._demodB.IQ))
        percycle = len(self._demodA.IQ) / cycles
        IQA = self._demodA.IQ.reshape((cycles, percycle))

        # No weighting functions
        if self._Iweight is None:
            IQ_rel = np.average(IQA, 1) * phase_corr

        # Correct each trace, see numpy broadcasting
        else:
            IQA = (IQA.T * phase_corr).T
            IQ_rel = np.inner(np.real(IQA), self._Iweight) + 1j * np.inner(np.imag(IQA), self._Qweight)

        return IQ_rel

    def take_experiment(self, acqtimeout=None, avg_buf=None, shots_data=None, shots_avg=1):
        if acqtimeout is None:
            acqtimeout = self.get_timeout()
        IQ_sum = None
        IQ_shots_sum = None
        cycles = self._cycles
        navg = self.get_naverages()

        cyclereps = self._cyclereps
        recperbuf = self.get_nrecperbuf()
        totrec = self.get_ntotal_rec()

        numbufs = totrec / recperbuf
        shots_IQ = np.zeros(totrec, dtype=np.complex)

        i = 0
        century_count = 0
        s = 0
        update_averages = False
        while i < numbufs:
            avgs = i * cyclereps
            if avgs >= century_count * 100:
                logging.info('Acquiring %d', avgs)
                self.emit('capture-progress', avgs)
                update_averages = True
                century_count += 1

            buf = self.get_next_buffer(acqtimeout)
            IQ = self.get_IQ_rel(buf, cycles * cyclereps)
         
            if shots_data:
                shots_IQ[i*recperbuf:(i+1)*recperbuf]=IQ
                    
            if cyclereps > 1:
                IQ = IQ.reshape([cyclereps, cycles])
                IQ = IQ.sum(axis=0)
            if IQ_sum is not None:
                IQ_sum += IQ
            else:
                IQ_sum = IQ

            self._card.post_buffers(buf)
                
            if avg_buf and update_averages:
                self.update_averages(avg_buf, IQ_sum, (i + 1) * cyclereps)
                update_averages = False

            i += 1

        self.end_capture()
        if IQ_sum is None:
            return None

        if avg_buf:
            self.update_averages(avg_buf, IQ_sum, navg)
        if shots_data:
            shots_IQ=shots_IQ.reshape(totrec/(shots_avg*cycles), shots_avg, cycles)
            shots_IQ=np.average(shots_IQ, 1)
            shots_IQ=shots_IQ.reshape(1, totrec/shots_avg)
            shots_data[:] = self.convert_signal(shots_IQ)

        return self.convert_signal(IQ_sum / navg)
        
    def take_experiment_shots(self, acqtimeout=None, shotsIQ=None):
        if acqtimeout is None:
            acqtimeout = self.get_timeout()
        cycles = self._cycles
        cyclereps = self._cyclereps
        recperbuf = self.get_nrecperbuf()
        totrec = self.get_ntotal_rec()

        numbufs = totrec / recperbuf
        shots_IQ = np.zeros(totrec, dtype=np.complex)

        i = 0
        while i < numbufs:
            buf = self.get_next_buffer(acqtimeout)

            IQ = self.get_IQ_rel(buf, cycles * cyclereps)
            shots_IQ[i*recperbuf:(i+1)*recperbuf]=IQ
            self._card.post_buffers(buf)
            
            i += 1            
            
        shotsIQ = shotsIQ.reshape(totrec/cycles, cycles)
        
        return self.convert_signal(shotsIQ)
    
    
    
    def setup_hist(self, N, hist_buf=None):
        '''
        Setup histogram measurement for <N> shots.
        '''
        self.end_capture()
        nbufs, N = self.break_records(N, nsamples=self.get_nsamples())
        self.set_nrecperbuf(N)
        if nbufs != 1:
            self.set_nbuffers(min(4, nbufs))
        else:
            self.set_nbuffers(1)
        self.set_ntotal_rec(nbufs * N)
        self.allocate_buffers()
        periods = self.get_nsamples() / self.get_if_period()
        self.set_demod(avg_periods=periods)
        self.prepare_capture()
        self._card.post_buffers(self._bufs)
        if hist_buf is None:
            hist_buf = np.zeros((nbufs*N,), dtype=np.complex)
        self._hist_buf = hist_buf
        self.start_capture()

    def take_hist(self, acqtimeout=None):
        if acqtimeout is None:
            acqtimeout = self.get_timeout()

        i = 0
        cycles = self.get_nrecperbuf()
        ntot = self.get_ntotal_rec()
        nbufs = ntot / cycles
        while i < nbufs:
            if (i % 10) == 0:
                logging.info('Acquiring %d', i*cycles)
                self.emit('capture-progress', i*cycles)
            buf = self.get_next_buffer(acqtimeout)
            self._hist_buf[i*cycles:(i+1)*cycles] = self.get_IQ_rel(buf, cycles)
            self._card.post_buffers(buf)
            i += 1

        self.end_capture()

        return self._hist_buf

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    alz = Instrument.test(Alazar_Daemon)

    win = np.zeros(64)
    win[10:40] = 1
    np.save('win.npy', win)

    alz.set_nsamples(20*64)
    alz.set_ch1_range('40mV')
    alz.set_ch2_range('100mV')
    alz.set_ch1_coupling('AC')
    alz.set_ch2_coupling('AC')
    alz.set_clock_source('EXT10M')
    alz.set_sample_rate('1GEXT10')
    alz.set_engJ_trig_src('EXT')
    alz.set_engJ_trig_lvl(128+5)
    alz.set_weight_func('win.npy')

    alz.setup_clock()
    alz.setup_channels()
    alz.setup_trigger()

    LEN = 121
    nsamples = alz.get_nsamples()
    tracelen = nsamples / alz.get_if_period()

    plt.figure()
    alz.setup_shots(1)
    shots = alz.take_raw_shots()
    plt.plot(np.abs(shots[:nsamples]), label='A')
    plt.plot(np.abs(shots[nsamples:2*nsamples]), label='B')
    plt.legend()

    plt.figure()
    alz.setup_shots(1)
    shots = alz.take_demod_shots()
    plt.plot(shots)

    plt.figure()
    alz.setup_avg_shot(1000)
    buf = alz.take_avg_shot()
    plt.plot(buf)

    plt.figure()
    alz.set_naverages(500)
    alz.setup_experiment(100)
    buf = alz.take_experiment()
    plt.plot(buf)

    plt.figure()
    alz.setup_hist(100)
    vals = alz.take_hist()
    plt.scatter(np.real(vals), np.imag(vals))

    plt.show()
