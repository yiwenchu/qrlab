# Coincidence.py
# Reinier Heeres <reinier@heeres.eu>, 2011
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

from instrument import Instrument
import types
import pyvisa.vpp43 as vpp43
import time
import logging
from lib import visafunc

class Coincidence(Instrument):
    '''
    '''

    def __init__(self, name, address):
        Instrument.__init__(self, name, tags=['physical'])

        # Add parameters
        self.add_parameter('measurement_time',
            flags=Instrument.FLAG_SET|Instrument.FLAG_SOFTGET,
            type=types.IntType, minval=1, maxval=65535, units='dsec')

        self._open_serial_connection()

        self.add_function('get_all')

        if reset:
            self.reset()
        else:
            self.get_all()

    def __del__(self):
        self._close_serial_connection()

    def _open_serial_connection(self):
        self._session = vpp43.open_default_resource_manager()
        self._vi = vpp43.open(self._session, self._address)

        vpp43.set_attribute(self._vi, vpp43.VI_ATTR_ASRL_BAUD, 115200)
        vpp43.set_attribute(self._vi, vpp43.VI_ATTR_ASRL_DATA_BITS, 8)
        vpp43.set_attribute(self._vi, vpp43.VI_ATTR_ASRL_STOP_BITS,
            vpp43.VI_ASRL_STOP_ONE)
        vpp43.set_attribute(self._vi, vpp43.VI_ATTR_ASRL_PARITY,
            vpp43.VI_ASRL_PAR_ODD)
        vpp43.set_attribute(self._vi, vpp43.VI_ATTR_ASRL_END_IN,
            vpp43.VI_ASRL_END_NONE)

    def _close_serial_connection(self):
        vpp43.close(self._vi)

    def reset(self):
        self.set_measurement_time(0)
        time.sleep(0.1)
        self.set_measurement_time(1)
        time.sleep(0.1)

    def _short_to_bytes(self, ):
        dataH = int(bytevalue/256)
        dataL = bytevalue - dataH*256
        return (dataH, dataL)

    def do_set_measurement_time(self, dt):
        (DataH, DataL) = self._short_to_bytes(dt)
        message = "%c%c" % (DataH, DataL)
        vpp43.write(self._vi, message)

    def start(self):
        mt = self.get_measurement_time()
        self.do_set_measurement_time(0)
        time.sleep(0.1)
        self.do_set_measurement_time(mt)

    def read(self):
        data = visafunc.read_all(self._vi)
        ret = ''
        for c in data:
            ret += '%02x,' % c
        print 'Read %d bytes' % (len(data), )
        print 'Data: %s' % (ret, )

