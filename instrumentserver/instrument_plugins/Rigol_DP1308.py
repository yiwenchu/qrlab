# Rigol_DP1308.py class, to perform the communication between the Wrapper and the device
# vishal ranjan, 2012
# Gijs de Lange, 2012
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
import visa
import types
import logging
import numpy
from time import sleep

class Rigol_DP1308(Instrument):
    '''
    This is the python driver for the Rigol current source

    Usage:
    Initialize with
    <name> = instruments.create('name', 'RS_SMB100', address='<GPIB address>',
        reset=<bool>)
    '''

    def __init__(self, name, address, reset=False):
        '''
        Initializes the Rigol, and communicates with the wrapper.

        Input:
            name (string)    : name of the instrument
            address (string) : GPIB address
            reset (bool)     : resets to default values, default=false

        Output:
            None
        '''

	self._Ires = 0.01

        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['physical'])

        self._address = address
        self._visainstrument = visa.instrument(self._address, timeout = 2)
        print ' Rigol timeout set to: %s s'%self._visainstrument.timeout

	# add parameters
        self.add_parameter('channel', type=types.StringType,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)

        self.add_parameter('status', type=types.StringType,
            flags=Instrument.FLAG_GETSET | Instrument.FLAG_GET_AFTER_SET)

        self.add_parameter('current', type=types.FloatType,
                flags=Instrument.FLAG_GETSET, units = 'Amps')




        self.add_function('reset')
        self.add_function('get_all')

        if reset:
            self.reset()
        else:
            self.get_all()


	    # add functions here to enable OVP and OCP

    def reset(self):
        '''
        Resets the instrument to default values

        Input:
            None

        Output:
            None
        '''
        logging.info(__name__ + ' : Resetting instrument')
        self._visainstrument.write('*RST')



    def get_all(self):
        '''
        Reads all implemented parameters from the instrument,
        and updates the wrapper.

        Input:
            None

        Output:
            None
        '''
        logging.info(__name__ + ' : reading all settings from instrument')
        self.get_channel()
        self.get_status()
        self.get_current()

    def do_set_channel(self,channel):
	'''
	selects an operating channel
	input: P6V, P25V, N25V
	output: None
	'''

        if channel == 'P6V':
	    self._visainstrument.write('INST:SELE P6V')
        elif channel == 'P25V':
            self._visainstrument.write('INST:SELE P25V')
        elif channel == 'N25V':
            self._visainstrument.write('INST:SELE N25V')
        else:
            raise ValueError('Channel %s does not exist' %channel)


    def do_get_channel(self):
	'''
        returns the operating channel
	'''

	a= self._visainstrument.ask('INST:SELE?')
	if a == 1:
	    return 'P6V'
	if a == 2:
	    return 'P25V'
	if a == 1:
	    return 'N25V'


    def do_set_status(self,value):
	'''
	takes value = on/off
        sets the status = on/off
	for the operating channel
	'''

	channel = self.get_channel()
        a= self._visainstrument.write('OUTP:STAT %s, %s' %(channel,value))

    def do_get_status(self):
	'''
	returns the status = on/off
	for the operating channel
	'''

	channel = self.get_channel()
        return self._visainstrument.ask('OUTP:STAT? %s' %channel)



    def get_visains(self):
	return self._visainstrument


    def do_set_current(self,I):
	'''
	sets the current on the selected channel
	mind the max and min currents

	'''
	self.ramp_current(I)

    def do_get_current(self,I):
	'''
	sets the current on the selected channel
	mind the max and min currents

	'''
	return self._visainstrument.ask('SOUR:CURR:LEV:IMM:AMPL ?')

    def ramp_current(self, value, ramp_speed = 0.1):
        '''
        ramps the to value
        ramp_speed is rampspeed in Amps/second
        '''
	cur_val = self.get_current()
        dI = 1.*value - cur_val
	Ires = self._Ires
        n_steps = abs(int(dI/Ires))

        for k in range(n_steps):
	    self._visainstrument.write('SOUR:CURR:LEV:IMM:AMPL %s' % (1.*k)/n_steps*dI+cur_val)
            qt.msleep(Ires/ramp_speed)

        return 'A current of %s Amps has been set on %s channel' \
			%(self.get_current(),self.get_channel())







