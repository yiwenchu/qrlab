# FPGA_Yngwie.py, instrument driver for Yngwie FPGA card
# Reinier Heeres 2014

import os
import sys
import inspect
import time
import ctypes
import types
import numpy as np
from instrumentserver.instrument import Instrument
import logging

# Add paths for dependencies.
# srcdir = os.path.split(os.path.abspath(inspect.getsourcefile(lambda _: None)))[0]
# basedir = '\\'.join(srcdir.split('\\')[:-2])
# sys.path.append(os.path.join(basedir, 'Yngwie\\Python\\Core'))

import YngwieInterface

INI_DEFAULT = '''
adcTestEnable=0
Alerts=65535
ClockFrequency=1000
dacTestEnable=0
dacTestFrequency=20
DumpEnabled=0
DumpEntireVelo=0
DumpPath
ExternalClock=1
ExternalReference=1
ReferenceFrequency=1000
Rx.EdgeMode=1
Rx.EnabledChannels=3
Rx.ExternalTrigger=1
Rx.FrameMode=0
Rx.FrameSize=65536
Rx.SampleRate=1000
SeparatePackets=0
TimerInterval=1000
TriggerDelay=1
Tx.EdgeMode=1
Tx.ExternalTrigger=1
Tx.FrameMode=0
Tx.FrameSize=8192
VeloPacketSize=65336
'''

class Yngwie_FPGA(Instrument):

    def __init__(self, name, boardid=None, open=True, **kwargs):
        super(Yngwie_FPGA, self).__init__(name)

        if boardid is None:
            raise Exception('Yngwie driver needs a boardid')
        self._target = int(boardid)

        self.yng = None
        self._ssb_freq = {}
        self._ssb_theta = {}
        self._ssb_ratio = {}

        self._noutputs = kwargs.pop('noutputs', 4)
        self._nmodes = kwargs.pop('nmodes', 2)

        self.add_parameter('logic_version', type=str,
            option_list=('AXX', 'BXX'), value='AXX',
            flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET)
        self.add_parameter('noutputs', type=types.IntType,
            option_list=(2,4), value=self._noutputs,
            flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET)
        self.add_parameter('nmodes', type=types.IntType,
            options_list=(1,2,4), value=self._nmodes,
            flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET)
        self.add_parameter('reshape_mode', type=types.StringType,
            option_list=('full', 'none', 'mixer'),
            flags=Instrument.FLAG_GETSET)
        self.add_parameter('dot_product_mode', type=types.StringType,
            option_list=('dot product', "pass both", "dup ch0", "dup ch1"),
            flags=Instrument.FLAG_GETSET)

        modes = [i for i in range(self._nmodes)]
        self.add_parameter('ssbfreq', type=types.FloatType, channels=modes,
            value=0, gui_group='ssb',
            flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET)
        self.add_parameter('ssbtheta', type=types.FloatType, channels=modes,
            min=-180, max=180, units='deg', format='%.03f',
            value=0, gui_group='ssb',
            flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET)
        self.add_parameter('ssbratio', type=types.FloatType, channels=modes,
            min=0, max=1.99, format='%.06f',
            value=1, gui_group='ssb',
            flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET)
        self.add_parameter('offset', type=types.TupleType, channels=(0, 1),
            gui_group='ssb',
            flags=Instrument.FLAG_GETSET)

        self.add_parameter('fillup_thresh', type=types.IntType,
            min=0, max=255, gui_group='internal',
            flags=Instrument.FLAG_GET)
        self.add_parameter('regulation_thresh', type=types.IntType,
            min=0, max=255, gui_group='internal',
            flags=Instrument.FLAG_GET)

        self.add_parameter('regulation_enabled', type=types.BooleanType,
            gui_group='internal', flags=Instrument.FLAG_GETSET)
        self.add_parameter('awg_enabled', type=types.BooleanType,
            gui_group='internal', flags=Instrument.FLAG_GETSET)
        self.add_parameter('external_trigger_enabled', type=types.BooleanType,
            gui_group='internal', flags=Instrument.FLAG_GETSET)

        self.add_parameter('dump_path', type=types.StringType,
            flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
            value=r'd:\data\fpga\exp_')
        self.add_parameter('run_status', type=types.IntType,
            flags=Instrument.FLAG_GET)
        self.add_parameter('unlimited', type=types.BooleanType,
            flags=Instrument.FLAG_GETSET)

        self.add_parameter('buffer_gen_width', type=types.IntType,
            flags=Instrument.FLAG_GETSET)
        self.add_parameter('buffer_gen_delay', type=types.IntType,
            flags=Instrument.FLAG_GETSET)

        self.add_parameter('delay_analog', type=types.IntType,
            gui_group='delays', flags=Instrument.FLAG_GETSET)
        self.add_parameter('delay_marker', type=types.IntType, channels=(0,1,2,3),
            gui_group='delays', flags=Instrument.FLAG_GETSET)
        self.add_parameter('rrec_generate', type=types.IntType,
            flags=Instrument.FLAG_GETSET,
            help='Get/set result record generation mask')
        self.add_parameter('buffer_mask', type=types.IntType, channels=(0,1,2,3),
            flags=Instrument.FLAG_GETSET|Instrument.FLAG_SOFTGET, value=15)
        self.add_parameter('demod_reset_mode', type=types.BooleanType,
                           value=False, flags=Instrument.FLAG_GETSET)

        self.add_function('stop')
        self.add_function('update_modes')

        if open:
            self.open()
            self.set(kwargs)

        if kwargs.pop('reset', False):
            self.reset()
        else:
            self.get_all()

    def write_ini_file(self, fn='yngwie.ini'):
        data = INI_DEFAULT

        if self.get_noutputs() == 2:
            data += 'Tx.EnabledChannels=5\nTx.SampleRate=1000\nfourOutputChannels=0'
        else:
            data += 'Tx.EnabledChannels=15\nTx.SampleRate=500\nfourOutputChannels=1'

        f = open(fn, 'w')
        f.write(data)
        f.close()

    def open(self):
        '''Open or re-open device.'''

        if self.yng:
            del self.yng
            logging.info('Reopening Yngwie...')
        else:
            logging.info('Opening Yngwie...')

        four_chans = self.get_noutputs() == 4
        kwargs = {
            'four_channels_enabled': four_chans,
            'four_channels_enable': four_chans,
        } # Thanks Nissim
        self.yng = YngwieInterface.YngwieInterface(self._target, **kwargs)
        print 'Setting dump path to %s' % (self.get_dump_path(),)
        self.yng.dump_path = self.get_dump_path()

        self.update_modes()

    def update_modes(self):
        '''
        Update analog mode settings on FPGA.
        '''

        nmodes = self.get_nmodes()
        if nmodes == 2:
            mul = 2
        else:
            mul = 1

        self.yng.AnalogModes.number = nmodes
        self.yng.AnalogModes.reshape_mode = self.get_reshape_mode()
        self.yng.AnalogModes.four_channels_enabled = (self.get_noutputs() == 4)

        for i in range(nmodes):
            freq = int(round(self._ssb_freq.get(i, 0)))
            theta = self._ssb_theta.get(i, 0)
            ratio = self._ssb_ratio.get(i, 1.0)
            self.yng.AnalogModes.load(mode_index=mul*i,
                    frequency=freq,
                    theta=theta,
                    ratio=ratio,
            )
        if self.get_logic_version() == 'BXX':
            # mask: bit 0 = mode 0, bit 1 = mode 1, ...
            # setup so that buffer 0 and 2 look at mode 0/1 and buffer 1 and 3 look at mode 2/3
            # combiner allows to "AND" with external function
            masks = [self.get('buffer_mask%d' % i) for i in range(4)]
            for i, mask in enumerate(masks):
                self.yng.AutoBuffers[i].mask = mask
                # self.yng.AutoBuffers[i].pattern = 2**32 - 1
                self.yng.AutoBuffers[i].combiner = 0
                # ab = self.yng.AutoBuffers[i]
                # print i, ab.mask, ab.pattern, ab.combiner
            self.do_set_buffer_gen_width(65535)

    def do_set_noutputs(self, v):
        self._noutputs = v
        self.open()

    def do_set_nmodes(self, v):
        self.yng.AnalogModes.number = v
        self._nmodes = v

    def do_set_reshape_mode(self, v):
        self.yng.AnalogModes.reshape_mode = v.lower()

    def do_get_reshape_mode(self):
        return self.yng.AnalogModes.reshape_mode

    def do_set_dot_product_mode(self, v):
        self.yng.dot_product_mode = v.lower()

    def do_get_dot_product_mode(self):
        return self.yng.dot_product_mode

    def do_set_offset(self, v, channel=None):
        if len(v) != 2:
            raise ValueError('Offset should be specified as 2 integers')
        setattr(self.yng.AnalogModes, 'offset%d' % channel, v)

    def do_get_offset(self, channel=None):
        return getattr(self.yng.AnalogModes, 'offset%d' % channel)

    def do_set_ssbfreq(self, v, channel=None):
        self._ssb_freq[channel] = v

    def do_set_ssbtheta(self, v, channel=None):
        self._ssb_theta[channel] = v

    def do_set_ssbratio(self, v, channel=None):
        self._ssb_ratio[channel] = v

    def do_get_fillup_thresh(self):
        return self.yng.fillup_threshold

    def do_set_fillup_thresh(self, v):
        self.yng.fillup_threshold = v

    def do_get_regulation_thresh(self):
        return self.yng.regulation_threshold

    def do_set_regulation_thresh(self, v):
        self.yng.regulation_threshold = v

    def do_set_dump_path(self, v):
        self.yng.dump_path = v

    def do_read_logic(self, address, bitrange=(31,0)):
        return self.yng.m_yng.ReadLogic(address, bitrange)

    def do_write_logic(self, address, val, bitrange=(31,0)):
        return self.yng.m_yng.WriteLogic(address, val, bitrange)

    def accept_stream(self, card_n, streamID, bytes_needed, file_size=None, first_file_size=None, stream_pattern=0xffffffff):
        # card_n supplied only to make accept_stream have the same signature
        # on Yngwie_FPGA and on BlackMamba_FPGA
        assert card_n == self._target
        print 'Accepting %s,%s,%s' % (streamID, bytes_needed, file_size)
        return self.yng.StreamRouter.accept(streamID, bytes_needed, file_size, first_file_size, stream_pattern)

    def do_set_buffer_gen_width(self, val):
        if self.get_logic_version() == 'BXX':
            # self.yng.AutoBuffers.buffer_hold = val
            for i in range(4):
                self.yng.AutoBuffers[i].pattern = val
        else:
            self.do_write_logic(0xc05, 13, [21,16])
            self.do_write_logic(0xc04, val)

    def do_get_buffer_gen_width(self):
        if self.get_logic_version() == 'BXX':
            return self.yng.AutoBuffers[0].pattern
            # return self.yng.AutoBuffers.buffer_hold
        return self.do_read_logic(0xc04)

    def do_set_buffer_gen_delay(self, val):
        if self.get_logic_version() == 'BXX':
            self.yng.AutoBuffers.buffer_delay = val
        else:
            self.do_write_logic(0xc05, 13, [21,16])
            self.do_write_logic(0xc05, val, [29, 24])

    def do_get_buffer_gen_delay(self):
        if self.get_logic_version() == 'BXX':
            return self.yng.AutoBuffers.buffer_delay
        return self.do_read_logic(0xc05, [29, 24])

    def do_set_rrec_generate(self, val):
        self.do_write_logic(0xc03, val)

    def do_get_rrec_generate(self):
        return self.do_read_logic(0xc03)

    def is_streaming(self):
        return self.yng.m_yng.IsStreaming()

    def do_get_regulation_enabled(self):
        return self.yng.regulation_enabled

    def do_set_regulation_enabled(self, v):
        self.yng.regulation_enabled = v

    def do_get_awg_enabled(self):
        return self.yng.awg_enabled

    def do_set_awg_enabled(self, v):
        self.yng.awg_enabled = v

    def do_get_external_trigger_enabled(self):
        return self.yng.external_trigger_enabled

    def do_set_external_trigger_enabled(self, v):
        self.yng.external_trigger_enabled = v

    def do_get_delay_analog(self):
        return self.yng.Delays.analog

    def do_set_delay_analog(self, v):
        self.yng.Delays.analog = v

    def do_set_delay_marker(self, v, channel=None):
        setattr(self.yng.Delays, 'marker%d'%channel, v)

    def do_get_delay_marker(self, channel=None):
        return getattr(self.yng.Delays, 'marker%d'%channel)

    def load_tables(self, file_prefix):
        file_prefix = file_prefix + '_B%d' % self._target
        print 'Yngwie: Loading tables from %s' % (file_prefix,)
        self.yng.regulation_enabled = True
        return self.yng.load_tables(file_prefix)

    def start(self):
        self.yng.m_yng.DumpSettings()
        self.yng.StreamRouter.dump()

        print 'Yngwie: Starting...'
        self.yng.start()
        time.sleep(0.5)
        self.set_awg_enabled(True)
        time.sleep(0.1)
        self.set_external_trigger_enabled(True)
        time.sleep(0.1)

    def stop(self):
        print 'Yngwie: Stopping...'
        self.yng.StreamRouter.dump()
        self.yng.stop()
        self.set_external_trigger_enabled(False)
        self.set_awg_enabled(False)

    def do_get_run_status(self):
        return self.yng.run_status()

    def do_get_unlimited(self):
        return self.yng.StreamRouter.unlimited

    def do_set_unlimited(self, v):
        self.yng.StreamRouter.unlimited = v

    def close(self):
        '''Clean-up yngwie (to prevent crashing).'''
        if not self.yng:
            return
        self.stop()
        del self.yng
        self.yng = None

    def read_registers(self, n):
        return self.yng.m_registers

    def check_trigger(self):
        return self.get_run_status() == 0

    def do_get_demod_reset_mode(self):
        return self.yng.demod_reset_mode

    def do_set_demod_reset_mode(self, val):
        self.yng.demod_reset_mode = val


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    yng = Instrument.test(Yngwie_FPGA)
