# Newport_ESPx00.py class, to perform the communication between the Wrapper and the device
# Martijn Schaafsma <qtlab@mcschaafsma.nl>, 2010, 2011
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
import logging
import numpy
import visa
import time as time

# TODO: adapt for rotation stage, but keep compatible with delay stage
#       add a function for relative movement

class Newport_ESPx00(Instrument):
  '''
  This is the driver class for the ESP stepper motor

  Usage:
    Initialize with
    <name> = instruments.create('<name>', 'Newport_ESPx00', address='<COM address>, numaxes=<int>, reset=<bool>')

    TODO: - Check if there is a difference in imposing a displacement or sending the stage to a certain location
          - Add the function "move(val[mm])" and scale this guy in a virtual instrument to ps

  '''
  def __init__(self, name, address, numaxes=1, reset=False):
    logging.info(__name__ + ' : Initializing instrument Newport ESP100')
    Instrument.__init__(self, name, tags=['physical'])

    self._numaxes = numaxes
    self._address = address
    self._visainstrument = visa.instrument(self._address)

    # Add functions
    self.add_function('init_default')
    self.add_function('define_home')

    # Add parameters
    self.add_parameter('position',tags=['sweep'],channels=(1,numaxes),channel_prefix='Axis%d_',
      flags=Instrument.FLAG_GETSET, units='mm', minval=-300, maxval=300, type=types.FloatType)
    self.add_parameter('move_relative',tags=['sweep'],channels=(1,numaxes),channel_prefix='Axis%d_',
      flags=Instrument.FLAG_SET, units='mm', minval=-300, maxval=300, type=types.FloatType)
    self.add_parameter('ismoving',channels=(1,numaxes),channel_prefix='Axis%d_',
      flags=Instrument.FLAG_GET, type=types.StringType)

    self.init_default()

    if reset:
      self.init_default()

  def init_default(self):
    print "Initializing ESP100 ..."
    self._visainstrument.baud_rate=19200L
    self._visainstrument.clear()
#    self._visainstrument.write('1MO;WS\r')
#    self._visainstrument.write('1DH;WS\r')
#    self._visainstrument.write('1MT-;WS\r')
###    self._visainstrument.write('1PA-300;WS\r')
    print "Waiting for stage to have moved"
    while self.do_get_ismoving():
      print "  ... still moving ..."
      time.sleep(1)
#    self._visainstrument.write('1DH;WS\r')
    print "Finished initialization ESP100"
    self.get_position()

  def do_get_position(self, channel=1):
    return self._visainstrument.ask('%dPA?\r'%(channel))

  def define_home(self, position=0.0, channel=1):
    self._visainstrument.write('%dDH%f;WS\r'%(channel, position))

  def do_set_position(self, position, channel=1):
    self._visainstrument.write('%dPA+%f;WS\r'%(channel, position))

  def do_get_ismoving(self, channel=1):
    return not self._visainstrument.ask('%dMD?'%(channel)).strip() == '1'

  def do_set_move_relative(self, position, channel=1):
    self._visainstrument.write('%dPR+%f;WS\r'%(channel, position))
    return self.get_position()
