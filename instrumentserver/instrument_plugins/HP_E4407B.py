import types
import logging

import numpy as np

from instrument import Instrument
from visainstrument import SCPI_Instrument


class HP_E4407B(SCPI_Instrument):
    '''
    This is the driver for the HP E4407B Spectrum Analyzer

    Usage:
    Initialize with
    <name> = instruments.create('<name>', 'HP_E4407B', address='<GBIP address>, reset=<bool>')
    '''

    def __init__(self, name, address, **kwargs):
        '''
        Initializes the HP_E4407B, and communicates with the wrapper.

        Input:
          name (string)    : name of the instrument
          address (string) : GPIB address
          reset (bool)     : resets to default values, default=False
        '''
        logging.info(__name__ + ' : Initializing instrument Agilent_E5071C')
        super(HP_E4407B, self).__init__(name, address)

        self.add_scpi_parameter("start_freq", "SENS:FREQ:STAR", "%d", units="Hz", type=types.FloatType)
        self.add_scpi_parameter('stop_freq', "SENS:FREQ:STOP", "%d", units="Hz", type=types.FloatType)
        self.add_scpi_parameter('center_freq', "SENS:FREQ:CENT", "%d", units="Hz", type=types.FloatType)
        self.add_scpi_parameter('span', "SENS:FREQ:SPAN", "%d", units="Hz", type=types.FloatType)
        self.add_scpi_parameter('peak_loc', "CALC:MARK:X", "%d", units="Hz", type=types.FloatType)
        self.add_scpi_parameter('peak_height', "CALC:MARK:Y", "%d", units="dB", type=types.FloatType)
        self.add_scpi_parameter('if_bandwidth', "SENS:BAND", "%d", units="Hz", type=types.FloatType)
        self.add_scpi_parameter('points', "SENS:SWE:POIN", "%d", type=types.IntType)
        self.add_scpi_parameter('average_factor', "SENS:AVER:COUN", "%d", type=types.IntType)
        self.add_scpi_parameter('error', "SYST:ERR", "%s", type=types.StringType, flags=Instrument.FLAG_GET)
        self.add_scpi_parameter('averaging_state', 'SENS:AVER', '%i', type=bool,
                                flags=Instrument.FLAG_GETSET)
        self.add_scpi_parameter('sweep_time', 'SENS:SWE:TIME', '%.8f', units="s", type=types.FloatType,
                                flags=Instrument.FLAG_GET)
        self._instrument_state_file = None
        self.set(kwargs)

    def do_get_data(self):
        return np.array((self.do_get_xaxis(),self.do_get_yaxes()))

    def do_get_xaxis(self):
        return np.linspace(self.get_start_freq(),self.get_stop_freq(),self.get_points())
        # return np.array(map(float, self.ask('CALC:DATA:XAXis?', timeout=0.1).split(',')))

    def do_get_yaxes(self):
        strdata = self.ask('TRAC? TRACE1', timeout=0.1)
        data = np.array(map(float, strdata.split(',')))
        return data
        
    def center_peak(self):
        self.write('CALC:MARK:MAX')
        self.write('CALC:MARK:CENT')
        return self.get_peak_loc()
        
    def single_meas(self):
        self.write('INIT:IMM')
        self.ask('*OPC?')
        return True
        
    def zoom(self,factor):
        current_span = self.get_span()
        self.set_span(current_span/factor)
        return self.get_span()

    def reset(self):
        self.write('*RST')

    def opc(self):
        return self.ask('*OPC?')

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)
    Instrument.test(HP_E4407B)
    # vna.do_get_data()
