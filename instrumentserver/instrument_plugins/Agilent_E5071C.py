import types
import logging

import numpy as np

from instrument import Instrument
from visainstrument import SCPI_Instrument


class Agilent_E5071C(SCPI_Instrument):
    '''
    This is the driver for the Agilent E5071C Vector Network Analyzer

    Usage:
    Initialize with
    <name> = instruments.create('<name>', 'Agilent_E8257D', address='<GBIP address>, reset=<bool>')
    '''

    def __init__(self, name, address, **kwargs):
        '''
        Initializes the Agilent_E5071C, and communicates with the wrapper.

        Input:
          name (string)    : name of the instrument
          address (string) : GPIB address
          reset (bool)     : resets to default values, default=False
        '''
        logging.info(__name__ + ' : Initializing instrument Agilent_E5071C')
        super(Agilent_E5071C, self).__init__(name, address)

        self.add_scpi_parameter("start_freq", "SENS:FREQ:STAR", "%d", units="Hz", type=types.FloatType, gui_group='sweep')
        self.add_scpi_parameter('stop_freq', "SENS:FREQ:STOP", "%d", units="Hz", type=types.FloatType, gui_group='sweep')
        self.add_scpi_parameter('correction', "SENS:CORR:STAT", '%i', type=bool, flags=Instrument.FLAG_GETSET, gui_group='calibration')
        self.add_scpi_parameter('center_freq', "SENS:FREQ:CENT", "%d", units="Hz", type=types.FloatType, gui_group='sweep')
        self.add_scpi_parameter('span', "SENS:FREQ:SPAN", "%d", units="Hz", type=types.FloatType, gui_group='sweep')
        self.add_scpi_parameter('electrical_delay', "CALC:SEL:CORR:EDEL:TIME", "%.4e", units="s", type=types.FloatType, gui_group='calibration')
        self.add_scpi_parameter('phase_offset', "CALC:SEL:CORR:OFFS:PHAS", "%.6f", units="deg", type=types.FloatType, gui_group='calibration')
        self.add_scpi_parameter('if_bandwidth', "SENS:BAND", "%d", units="Hz", type=types.FloatType)
        self.add_scpi_parameter('power', "SOUR:POW", "%.2f", units="dBm", type=types.FloatType)
        self.add_scpi_parameter('points', "SENS:SWE:POIN", "%d", type=types.IntType, gui_group='sweep')
        self.add_scpi_parameter('average_factor', "SENS:AVER:COUN", "%d", type=types.IntType, gui_group='averaging')
        self.add_scpi_parameter('error', "SYST:ERR", "%s", type=types.StringType, flags=Instrument.FLAG_GET)
        self.add_scpi_parameter('sweep_time', 'SENS:SWE:TIME:DATA', '%.8f', units="s", type=types.FloatType,
                                flags=Instrument.FLAG_GET, gui_group='sweep')
        self.add_scpi_parameter('segment_sweep_time', 'SENS:SEGM:SWE:TIME:DATA', '%.8f', units="s", type=types.FloatType,
                                flags=Instrument.FLAG_GET, gui_group='sweep')
        self.add_scpi_parameter('measurement', 'CALC:PAR:DEF', '%s',
                                type=types.StringType, flags=Instrument.FLAG_GETSET,
                                format_map={"S11": "S11",
                                       "S12": "S12",
                                       "S21": "S21",
                                       "S22": "S22"})
        self.add_scpi_parameter('format', 'CALC:FORM', '%s',
                                type=types.StringType, flags=Instrument.FLAG_GETSET,
                                format_map={"MLOG": "log mag",
                                       "PHAS": "phase",
                                       "GDEL": "group delay",
                                       "SLIN": "Smith linear",
                                       "SLOG": "Smith log/phase",
                                       "SCOM": "Smith Re/Im",
                                       "SMIT": "Smith R+jX",
                                       "SADM": "Smith G+jB",
                                       "PLIN": "polar lin",
                                       "PLOG": "polar log",
                                       "MLIN": "linear mag",
                                       "SWR": "VSWR",
                                       "REAL": "real",
                                       "IMAG": "imaginary",
                                       "UPH": "expanded phase",
                                       "PPH": "positive phase"})
        self.add_scpi_parameter('trigger_source', 'TRIG:SOUR', '%s',
                           type=types.StringType, flags=Instrument.FLAG_GETSET,
                           format_map={"INT": "internal",
                                       "EXT": "external",
                                       "MAN": "manual",
                                       "BUS": "external bus"})
        self.add_scpi_parameter('instrument_state_data', 'MMEM:STOR:STYP', '%s',
                           type=types.StringType, flags=Instrument.FLAG_GETSET,
                           format_map={"STAT": "measurement conditions only",
                                       "CST": "+ calibration",
                                       "DST": "+ data",
                                       "CDST": "+ calibration + data"},
                                       gui_group='calibration')
        self.add_scpi_parameter('sweep_type', 'SENS:SWE:TYPE', '%s',
                           type=types.StringType, flags=Instrument.FLAG_GETSET,
                           format_map={"LIN": "linear",
                                       "LOG": "logarithmic",
                                       "SEGM": "segment",
                                       "POW": "power"},
                                       gui_group='sweep')
        self.add_scpi_parameter('averaging_state', 'SENS:AVER', '%i', type=bool,
                                flags=Instrument.FLAG_GETSET, gui_group='averaging')
        self.add_scpi_parameter('averaging_trigger', 'TRIG:AVER', '%i', type=bool,
                                flags=Instrument.FLAG_GETSET, gui_group='averaging')
        self.add_scpi_parameter('smoothing', 'CALC:SMO', '%i', type=bool,
                                flags=Instrument.FLAG_GETSET, gui_group='averaging')
#        self.add_scpi_parameter('arbitrary_segments', 'SENS:SEGM:ARB', '%s',
#                           type=types.StringType, flags=Instrument.FLAG_GETSET,
#                           format_map={"1": "allowed",
#                                       "0": "disallowed"}) # requires firmware upgrade
        self.add_scpi_parameter('clock_reference', 'SENS:ROSC:SOUR', '%s',
                           type=types.StringType, flags=Instrument.FLAG_GET,
                           format_map={"INT": "internal",
                                       "EXT": "external"})
        self.add_parameter('instrument_state_file', type=types.StringType,
                           flags=Instrument.FLAG_GETSET)
        self.add_function("save_state")
        self.add_function("load_state")
        self.add_function("autoscale")
        self.add_function("track_L")
        self.add_function("track_R")
        self.add_function("find_min")
        self.add_function("set_cent")
        self._instrument_state_file = None
        self.set(kwargs)

    def do_enable_averaging(self, enable=True):
        s = 'ON' if enable else 'OFF'
        self.set_averaging_state(s)
        self.set_averaging_trigger(s)

    def do_get_data(self, fmt='PLOG', opc=False, trig_each_avg=False):
        prev_fmt = self.get_format()
        if opc:
            prev_trig = self.get_trigger_source()
            prev_avg_trig = self.get_averaging_trigger()
            self.set_trigger_source('BUS')
            self.write('INIT:CONT ON')
            if trig_each_avg: # breaks up long measurements
                self.set_averaging_trigger(0)
                avg_steps = self.get_average_factor() if bool(self.get_averaging_state()) else 1
                for a in np.arange(avg_steps):
                    self.trigger()
                    self.opc() # wait for completion
            else:
                self.set_averaging_trigger(1)
                self.trigger()
                self.opc() # wait for completion
        self.set_format(fmt)
        data = self.do_get_yaxes()
        self.set_format(prev_fmt)
        if opc:
            self.set_trigger_source(prev_trig)
            self.set_averaging_trigger(prev_avg_trig)
        return data

    def do_get_xaxis(self):
        return np.array(map(float, self.ask('CALC:DATA:XAXis?', timeout=0.1).split(',')))

    def do_get_yaxes(self):
        strdata = self.ask('CALC:DATA:FDATa?', timeout=0.1)
        data = np.array(map(float, strdata.split(',')))
        data = data.reshape((len(data)/2, 2))
        return data.transpose() # mags, phases

    def reset(self):
        self.write('*RST')

    def opc(self):
        return self.ask('*OPC?')

    def trigger(self):
        self.write('TRIG:SING')

    def do_multipoint_sweep(self, start, stop, step):
        n_points = self.get_points()
        span = n_points * step
        for start in np.arange(start, stop, span):
            self.set_start_freq(start)
            self.set_stop_freq(start + span)
            yield self.do_get_data()

    def do_power_sweep(self, start, stop, step):
        for i, power in enumerate(np.arange(start, stop, step)):
            self.set_power(power)
            yield self.do_get_data()

    def do_set_instrument_state_file(self, filename):
        self._instrument_state_file = filename

    def do_get_instrument_state_file(self):
        return self._instrument_state_file

    def save_state(self, sfile=None):
        if sfile:
            self._instrument_state_file = sfile
        if self._instrument_state_file:
            self.write('MMEM:STOR "%s.sta"' % self._instrument_state_file) # maybe want " or "" around %s

    def load_state(self, sfile=None):
        if sfile:
            self._instrument_state_file = sfile
        if self._instrument_state_file:
            self.write('MMEM:LOAD "%s.sta"' % self._instrument_state_file)
        self.get_all()

    def autoscale(self):
        self.write('DISP:WIND:TRAC:Y:AUTO')

    def find_min(self):
        self.write('CALC:MARK4 ON')
        self.write('CALC:MARK4:FUNC:TYPE MIN')
        self.write('CALC:MARK4:FUNC:EXEC')
        
    def set_cent(self):
        self.write('CALC:MARK4:SET CENT')

    def track_L(self):
        span = self.get_span()
        center = self.get_center_freq()
        self.set_center_freq(center-span/4)
        
    def track_R(self):
        span = self.get_span()
        center = self.get_center_freq()
        self.set_center_freq(center+span/4)

    def load_segment_table(self, table, transpose=False):
        # table should be of the form (array of starts, array of stops, array of points)
        header = '6,0,0,0,0,0,0,%d,'%(np.size(table[0]))
        starts = ['%.1f'%i for i in table[0]]
        stops = ['%.1f'%i for i in table[1]]
        points = ['%i'%i for i in table[2]]
        if transpose:
            indata = (starts,stops,points)
        else:
            indata = zip(*(starts,stops,points))
        flat = [a for b in indata for a in b]
        data = ','.join(flat)
        self.write('SENS:SEGM:DATA %s%s' % (header,data))

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    Instrument.test(Agilent_E5071C)
    # vna.do_get_data()
