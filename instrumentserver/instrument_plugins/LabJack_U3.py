# LabJack U3 driver
# Hannes Bernien <hagun83@gmail.com>, 2013
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
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  US
#
#
from instrument import Instrument

from measurement.hardware.Labjack.src import u3, LabJackPython #python commands for LJ, you should download this from labjack.com
import struct
import types
import logging
import numpy
import time
import os
import qt
from lib import config

class LabJack_U3(Instrument):

    #####configuration of the LabJack
    DAC_reg = 5000 #for the communication to the "intrinsic" dac
    sclPins = [4,6] # number of LJTDAC modules that are connected to the labjack are
                    # given by the number of elements and the address by the values
                    # connected to FIO4, FIO6
    EEPROM_ADDRESS = 0x50
    DAC_ADDRESS = 0x12

    def __init__(self, name):

        logging.info(__name__ + ' : Initializing instrument')
        Instrument.__init__(self, name, tags=['physical'])

        self._LJ = u3.U3()

        self._LJ.writeRegister(50590, 15) #Setting FIO0-3 to analog, and the rest to digital...
                                          #should be changed if FIO0-3 should be used for sth else
        self.add_parameter('bipolar_dac',
                flags=Instrument.FLAG_GETSET,
                type=types.FloatType,
                units='V',
                minval=-10., maxval=10.,
                channels=(0,1,2,3),
                doc='+/-10V dac with 14 bit resolution')

        self.bipolar_dac_values=[0.0,0.0,0.0,0.0]

        self.add_parameter('dac',
             flags=Instrument.FLAG_GETSET,
             type=types.FloatType,
             units='V',
             minval=0.0,maxval=5.0,
             channels=(0,1),
             doc='0-5V dac with 10 bit resolution')

        self.add_parameter('analog_in',
                flags=Instrument.FLAG_GET,
                type=types.FloatType,
                channels=(0,1,2,3),
                doc='0-2.4V anolog input with 12 bit resolution')

        self.dac_modules = {}

        self.getCalConstants()

        cfg_fn = os.path.abspath(
                os.path.join(qt.config['cfg_path'], name+'.cfg'))
        if not os.path.exists(cfg_fn):
            _f = open(cfg_fn, 'w')
            _f.write('')
            _f.close()

        self._parlist = ['bipolar_dac0','bipolar_dac1','bipolar_dac2','bipolar_dac3']
        self.ins_cfg = config.Config(cfg_fn)
        self.load_cfg()
        self.save_cfg()

    def get_all(self):
        for n in self._parlist:
            self.get(n)

    def load_cfg(self):
        params_from_cfg = self.ins_cfg.get_all()
        for p in params_from_cfg:
            if p in self._parlist:
                self.set(p, value=self.ins_cfg.get(p))

    def save_cfg(self):
        for param in self._parlist:
            value = self.get(param)
            self.ins_cfg[param] = value

    def toDouble(self, buffer):
        """
        Name: toDouble(buffer)
        Args: buffer, an array with 8 bytes
        Desc: Converts the 8 byte array into a floating point number.
        """
        if type(buffer) == type(''):
            bufferStr = buffer[:8]
        else:
            bufferStr = ''.join(chr(x) for x in buffer[:8])
        dec, wh = struct.unpack('<Ii', bufferStr)
        return float(wh) + float(dec)/2**32

    def getCalConstants(self):
        """
        Name: getCalConstants()
        Desc: Loads or reloads the calibration constants for the LJTic-DAC
              modules and writes it in the module dictionary
        """
        for i,v in enumerate(self.sclPins):
            module = {}
            module['sclPin'] = v
            module['sdaPin'] = v+1
            ## getting calibration data from the LJTDAC
            data = self._LJ.i2c(self.EEPROM_ADDRESS, [64],
                    NumI2CBytesToReceive=36, SDAPinNum = module['sdaPin'],
                    SCLPinNum = module['sclPin'])

            response = data['I2CBytes']
            module['aSlope'] = self.toDouble(response[0:8])
            module['aOffset'] = self.toDouble(response[8:16])
            module['bSlope'] = self.toDouble(response[16:24])
            module['bOffset'] = self.toDouble(response[24:32])

            if 255 in response:
                print 'The calibration constants seem a little off. Please \
                        go into settings and make sure the pin numbers are \
                         correct and that the LJTickDAC is properly attached.'

            self.dac_modules['LJTDAC'+str(i)] = module

    def do_set_bipolar_dac(self, voltage, channel):
        module =  self.dac_modules['LJTDAC'+str(int(channel/2.0))]
        try:
            #print [48+channel%2, int(((voltage*module['aSlope'])+module['aOffset'])/256),
                         #int(((voltage*module['aSlope'])+module['aOffset'])%256)]
            self._LJ.i2c(self.DAC_ADDRESS,
                     [48+channel%2, int(((voltage*module['aSlope'])+module['aOffset'])/256),
                         int(((voltage*module['aSlope'])+module['aOffset'])%256)],
                     SDAPinNum = module['sdaPin'],
                     SCLPinNum = module['sclPin'])
            self.bipolar_dac_values[channel]=voltage

        except:
            print "I2C Error! Something went wrong when setting the LJTickDAC. Is the device detached?"
        self.save_cfg()

    def do_get_bipolar_dac(self, channel):
            return self.bipolar_dac_values[channel]

    def do_set_dac(self, voltage, channel):
         self._LJ.writeRegister(self.DAC_reg+channel*2,voltage)

    def do_get_dac(self, channel):
         self._LJ.readRegister(self.DAC_reg+channel*2)

    def do_get_analog_in(self, channel):
         voltage  = self._LJ.getAIN(channel)
         return voltage
