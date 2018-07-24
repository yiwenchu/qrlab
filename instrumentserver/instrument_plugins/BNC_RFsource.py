from instrument import Instrument
from visainstrument import SCPI_Instrument
import time
from math import pi

class BNC_RFsource(SCPI_Instrument):

    def __init__(self, name, address, **kwargs):
        super(BNC_RFsource, self).__init__(
            name, address=address, term_chars='\n', **kwargs
        )

        get = Instrument.FLAG_GET
        getset = Instrument.FLAG_GETSET

        self.add_scpi_parameter('rf_on', 'OUTP', '%d', type=bool, flags=getset)

        self.add_scpi_parameter('frequency', 'FREQ', '%.06f', type=float,
                                flags=getset, units='Hz', minval=1e5,
                                maxval=20e9, display_scale=6)

        self.add_scpi_parameter('lock_src', 'ROSC:SOUR', '%s', type=str,
                                flags=getset, option_list=('INT', 'EXT'))

        self.add_scpi_parameter('extref_freq', 'ROSC:EXT:FREQ', '%.03f', type=float,
                                flags=getset, minval=1e6, maxval=100e6)

        self.add_scpi_parameter('locked', 'ROSC:LOCK', '%d', type=bool, flags=get)

        self.add_scpi_parameter('phase', 'PHASE', '%.03f', type=float,
                                flags=getset, units='rad', minval=-pi, maxval=pi)

        self.add_function('lock')

if __name__ == '__main__':
    Instrument.test(BNC_RFsource)
