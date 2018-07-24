# AVS47-IB driver
# writen by Mohammad Shafiei
#    for any questions contact me using the following emails <m.shafiei@tudelft.nl> or <em.shafiei@gmail.com>

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
from time import sleep
import visa
import types
import logging

class AVS47IB(Instrument):
    '''
    This is the python driver for the AVS47-IB

    Usage:
    Initialize with
    <name> = instruments.create('name', 'AVS47IB', address='<GPIB address>',
        reset=<bool>)
    '''

    def __init__(self, name, address, reset=False):
        '''
        Input:
            name (string)    : name of the instrument
            address (string) : GPIB address
            reset (bool)     : resets to default values, default=false

        Output:
            None
        '''
        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['physical'])

        self._address = address
        self._visainstrument = visa.instrument(self._address)

        self.add_parameter('input', flags=Instrument.FLAG_GETSET, type=types.StringType)
        self.add_parameter('range', flags=Instrument.FLAG_GETSET, type=types.IntType)
        self.add_parameter('excitation', flags=Instrument.FLAG_GETSET, type=types.IntType)
        self.add_parameter('remoteStatus', flags=Instrument.FLAG_GETSET, type=types.IntType)
        self.add_parameter('display', flags=Instrument.FLAG_GETSET, type=types.IntType)
        self.add_parameter('channel', flags=Instrument.FLAG_GETSET, type=types.IntType)

        self.add_parameter('resistance', flags=Instrument.FLAG_GET, type=types.FloatType)

        self.add_function('reset')
        self.add_function('get_all')

        self.add_function('connectToAVS')


        if reset:
            self.reset()
        else:
            self.get_all()


    # Functions

    def get_all(self):
        '''
        Update all device's parameters
        Input:
            None
        Output:
            None
        '''

        for cnt in self.get_parameter_names():
            if cnt != 'resistance':
                self.get(cnt)

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
        self.get_all()

    def do_set_input(self,param):
        '''
        Set the input of  AVS
        Input:
            param(string)=
                'Z' :Zero
                'M' :Measure
                'C' :Calibrate
        Output:
            None
        '''

        paramDic={'z':0,'m':1,'c':2}
        logging.debug(__name__ + ' : setting the input parameter for the instrument')
        self.connectToAVS()
        self._visainstrument.write('INP%s' %paramDic[param.lower()])

    def do_get_input(self):
        '''
        Set the input of  AVS
        Input:
            None
        Output:
            Zero
            Measure
            Calibrate

        '''


        paramDic={0:'Zero',1:'Measure',2:'Calibrate'}
        logging.debug(__name__ + ' : getting the input parameter from the instrument')
        self.connectToAVS()
        return paramDic[int(self._visainstrument.ask('INP?'))]

    def do_set_range(self,param):
        '''
        Set the range parameter of  AVS
        Input:
            param(int)=
                0   :   no range
                1   :   2R
                2   :   20R
                3   :   200R
                4   :   2K
                5   :   20K
                6   :   200K
                7   :   2M
        Output:
            None
        '''
        logging.debug(__name__ + ' : setting the range parameter for the instrument')
        self.connectToAVS()
        self._visainstrument.write('RAN%s' %param)

    def do_get_range(self):
        '''
        Set the range parameter of  AVS
        Input:
            None
        Input:
            0   :   no range
            1   :   2R
            2   :   20R
            3   :   200R
            4   :   2K
            5   :   20K
            6   :   200K
            7   :   2M
        Output:
            None
        '''
        logging.debug(__name__ + ' : getting the range parameter for the instrument')
        self.connectToAVS()
        return self._visainstrument.ask('RAN?')

    def do_set_excitation(self,param):
        '''
        Set the excitation parameter of  AVS
        Input:
            param(int)=
                0   :   no excitation
                1   :   3muV
                2   :   10muV
                3   :   30muV
                4   :   100muV
                5   :   300muV
                6   :   1mV
                7   :   3mV
        Output:
            None
        '''
        logging.debug(__name__ + ' : setting the excitation parameter for the instrument')
        self.connectToAVS()
        self._visainstrument.write('EXC%s' %param)

    def do_get_excitation(self):
        '''
        Get the excitation parameter of  AVS
        Input:
            None
        Output:
            0   :   no excitation
            1   :   3muV
            2   :   10muV
            3   :   30muV
            4   :   100muV
            5   :   300muV
            6   :   1mV
            7   :   3mV

        '''
        logging.debug(__name__ + ' : setting the excitation parameter for the instrument')
        self.connectToAVS()
        return self._visainstrument.ask('EXC?')

    def do_set_remoteStatus(self,status=1):
        '''
        Set the remote/local status of the device
        Input:
            Status=
                1:remote
                0:local
        Output:
            True
        '''

        logging.debug(__name__ + ' : reading frequency from instrument')
        self.connectToAVS()
        self._visainstrument.write('REM%s'%status)
        return True

    def do_get_remoteStatus(self):
        '''
        Get the remote/local status of the device

        Input:
            None
        Output:
            Remote status
                0:  Local
                1:  Remote
        '''

        logging.debug(__name__ + ' : reading remote status from instrument')
        self.connectToAVS()
        return float(self._visainstrument.ask('REM?'))


    def do_set_display(self,param=0):
        '''
        Set the display parameter of  AVS
        Input:
            param(int)=
                0   :   R
                1   :   deltaR
                2   :   ADJ ref
                3...:   ... (look at the device front panel)
                7   :   530 set PT
        Output:
            None
        '''
        logging.debug(__name__ + ' : setting the display parameter for the instrument')
        self.connectToAVS()
        self._visainstrument.write('DIS%s' %param)

    def do_get_display(self):
        '''
        Set the range parameter of  AVS
        Input:
            None
        Output:
            0   :   R
            1   :   deltaR
            2   :   ADJ ref
            3...:   ... (look at the device front panel)
            7   :   530 set PT
        '''
        logging.debug(__name__ + ' : getting the display parameter from the instrument')
        self.connectToAVS()
        return self._visainstrument.ask('DIS?')

    def do_set_channel(self,ch):
        '''
        Set the channel of  AVS
        Input:
            channel: channel number 0 - 7
        Output:
            None
        '''
        logging.debug(__name__ + ' : setting the channel parameter from the instrument')
        self.connectToAVS()
        return self._visainstrument.write('MUX%s'%ch)

    def do_get_channel(self):
        '''
        get the current channel of  AVS
        Input:
            None
        Output:
            channel: channel number 0 - 7

        '''
        logging.debug(__name__ + ' : getting the current channel parameter from the instrument')
        self.connectToAVS()
        return self._visainstrument.ask('MUX?')

    def _do_get_resistance(self):
        '''
        Get the resistance from AVS from the current channel
        Input:
            None
        Output:
            Resistance value in Ohm
        '''
        logging.debug(__name__ + ' : reading frequency from instrument')
        self.connectToAVS()
        self._visainstrument.write('ADC' )
        return float(self._visainstrument.ask('RES?'))

    def connectToAVS(self):
        '''
        Connect to the device
        Input:
            None
        Output:
            None
        '''
        if self._visainstrument.ask('REM?')=='0':
            self._visainstrument.write('REM1')

